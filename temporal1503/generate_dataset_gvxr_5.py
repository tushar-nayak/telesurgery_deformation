import os
import glob
import json
import numpy as np
import PIL.Image
import trimesh
import cv2
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from gvxrPython3 import gvxr

# --- CONFIGURATION ---
INPUT_DIR = "output_sequential_second"
OUTPUT_DIR = "output_gvxr_5"
TEMP_MESH_FILE = "temp_geometry_scaled.stl" 
IMAGE_SIZE = 512

# GEOMETRY (Standard C-Arm in MM)
SOURCE_POS   = np.array([-600.0, 0.0, 0.0]) # Source is 600mm away
DETECTOR_POS = np.array([ 400.0, 0.0, 0.0]) # Detector is 400mm away
PIXEL_SZ     = 0.6 # 0.6mm pixel size = ~300mm FOV

class XRayDepthGenerator:
    def __init__(self, mesh_path, output_dir):
        self.output_dir = output_dir
        self.base_name = os.path.basename(mesh_path).replace("_mesh.stl", "")
        
        # 1. Load Raw Mesh
        self.mesh = trimesh.load(mesh_path, process=False)
        
        # 2. CENTER IT
        self.mesh.apply_translation(-self.mesh.bounding_box.centroid)
        
        # 3. FORCE SCALE (The Fix)
        # We calculate the current bounding box diagonal.
        # We force it to be ~150mm (typical size for a neurovascular cluster).
        # This removes any ambiguity about units (meters vs mm).
        extent = self.mesh.bounding_box.extents
        current_diag = np.linalg.norm(extent)
        
        target_size = 150.0 # mm
        scale_factor = target_size / current_diag
        
        self.mesh.apply_scale(scale_factor)
        
        # 4. Save this "Normalized" Mesh for gVXR
        self.mesh.export(TEMP_MESH_FILE)

    def process_views(self):
        results = []
        
        # --- GVXR PHYSICS ---
        gvxr.createOpenGLContext()
        gvxr.setSourcePosition(SOURCE_POS[0], SOURCE_POS[1], SOURCE_POS[2], "mm")
        gvxr.usePointSource()
        
        # 40 keV + Iron = Maximum Contrast (Pitch Black Shadows)
        gvxr.setMonoChromatic(0.04, "MeV", 1000) 
        
        gvxr.setDetectorPosition(DETECTOR_POS[0], DETECTOR_POS[1], DETECTOR_POS[2], "mm")
        gvxr.setDetectorUpVector(0, 0, -1)
        gvxr.setDetectorNumberOfPixels(IMAGE_SIZE, IMAGE_SIZE)
        gvxr.setDetectorPixelSize(PIXEL_SZ, PIXEL_SZ, "mm")

        # Load the SCALED mesh
        gvxr.loadMeshFile("vessel", TEMP_MESH_FILE, "mm")
        gvxr.setElement("vessel", "Fe") 
        gvxr.setDensity("vessel", 7.87, "g/cm3") 

        for angle in [0, 90]:
            tag = f"ang{angle}"
            
            # Rotate
            if angle != 0:
                gvxr.rotateNode("vessel", angle, 0, 1, 0)
            
            # --- X-RAY ---
            raw_xray = np.array(gvxr.computeXRayImage()).astype(np.float32)
            
            # DSA Contrast (White BG, Black Vessel)
            if np.max(raw_xray) > np.min(raw_xray):
                img_norm = (raw_xray - np.min(raw_xray)) / (np.max(raw_xray) - np.min(raw_xray))
                img_uint8 = (img_norm * 255).astype(np.uint8)
            else:
                img_uint8 = np.full_like(raw_xray, 255, dtype=np.uint8)

            xray_filename = f"{self.base_name}_{tag}.png"
            PIL.Image.fromarray(img_uint8).save(os.path.join(self.output_dir, xray_filename))
            
            # --- DEPTH ---
            depth_filename = f"{self.base_name}_{tag}_depth.png"
            self.compute_depth_map(angle, depth_filename)
            
            # Metadata
            f_pix = 1000.0 / PIXEL_SZ 
            K = [[f_pix, 0, IMAGE_SIZE/2], [0, f_pix, IMAGE_SIZE/2], [0, 0, 1]]
            
            results.append({
                "image_file": xray_filename,
                "depth_file": depth_filename,
                "mesh_source": f"{self.base_name}_mesh",
                "view_angle": angle,
                "projection_matrix": K
            })

            # Reset Rotation
            if angle != 0:
                gvxr.rotateNode("vessel", -angle, 0, 1, 0)
        
        return results

    def compute_depth_map(self, angle, filename):
        # Use the SAME scaled mesh
        mesh_copy = self.mesh.copy()
        
        rot = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0])
        mesh_copy.apply_transform(rot)
        
        half_w = (IMAGE_SIZE * PIXEL_SZ) / 2.0
        y = np.linspace(-half_w, half_w, IMAGE_SIZE)
        z = np.linspace(-half_w, half_w, IMAGE_SIZE)
        yy, zz = np.meshgrid(y, z)
        
        pixel_locs = np.stack([np.full_like(yy, DETECTOR_POS[0]), yy, zz], axis=-1).reshape(-1, 3)
        ray_origins = np.tile(SOURCE_POS, (len(pixel_locs), 1))
        ray_dirs = pixel_locs - ray_origins
        ray_dirs /= np.linalg.norm(ray_dirs, axis=1)[:, np.newaxis]
        
        locations, index_ray, _ = mesh_copy.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_dirs,
            multiple_hits=False
        )
        
        depth_img = np.zeros(IMAGE_SIZE * IMAGE_SIZE)
        if len(locations) > 0:
            dists = np.linalg.norm(locations - SOURCE_POS, axis=1)
            # Normalize 550mm - 650mm range (Target is at 0, Source at -600)
            dists_norm = (dists - 550.0) / (100.0) 
            dists_norm = np.clip(dists_norm, 0, 1)
            depth_img[index_ray] = dists_norm * 255
            
        depth_img = depth_img.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)
        PIL.Image.fromarray(depth_img).save(os.path.join(self.output_dir, filename))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    mesh_files = glob.glob(os.path.join(INPUT_DIR, "*_mesh.stl"))
    mesh_files.sort()
    
    print(f"Processing {len(mesh_files)} meshes (Auto-Scaled to 150mm)...")
    
    dataset_labels = {}
    
    for mesh_path in tqdm(mesh_files):
        try:
            gen = XRayDepthGenerator(mesh_path, OUTPUT_DIR)
            items = gen.process_views()
            for item in items:
                key = item.pop("image_file")
                dataset_labels[key] = item
        except Exception as e:
            print(f"Failed on {mesh_path}: {e}")
            
    if os.path.exists(TEMP_MESH_FILE):
        os.remove(TEMP_MESH_FILE)

    with open(os.path.join(OUTPUT_DIR, "dataset.json"), 'w') as f:
        json.dump(dataset_labels, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()