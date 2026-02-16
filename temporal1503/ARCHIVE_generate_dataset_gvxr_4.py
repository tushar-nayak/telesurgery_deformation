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
OUTPUT_DIR = "output_gvxr_4"
TEMP_MESH_FILE = "temp_geometry.stl" # Intermediate file for alignment
IMAGE_SIZE = 512

# Unified Geometry (All in MILLIMETERS)
# Source at -60cm, Detector at +40cm. Total 1m throw.
SOURCE_POS_MM   = np.array([-600.0, 0.0, 0.0]) 
DETECTOR_POS_MM = np.array([ 400.0, 0.0, 0.0]) 
# Detector Field of View: 300mm (30cm) to ensure we see the whole vessel
DETECTOR_SIZE_MM = 300.0 
PIXEL_SZ_MM = DETECTOR_SIZE_MM / IMAGE_SIZE

class XRayDepthGenerator:
    def __init__(self, mesh_path, output_dir):
        self.original_path = mesh_path
        self.output_dir = output_dir
        self.base_name = os.path.basename(mesh_path).replace("_mesh.stl", "")
        
        # 1. Load and Standardize Mesh (Trimesh)
        self.mesh = trimesh.load(self.original_path, process=False)
        
        # Center the mesh at (0,0,0)
        box_center = self.mesh.bounding_box.centroid
        self.mesh.apply_translation(-box_center)
        
        # Scale Check: If object is small (<50 units), assume it's CM and scale to MM
        # Neuro vessels are usually ~50-100mm.
        extents = self.mesh.bounding_box.extents
        if np.max(extents) < 50.0: 
            self.mesh.apply_scale(10.0) # Convert cm -> mm
            
        # 2. Save Standardized Mesh for gVXR
        # This ensures gVXR sees exactly the same geometry as our Ray Caster
        self.mesh.export(TEMP_MESH_FILE)

    def contrast_stretch(self, img_array):
        """Auto-contrast for DSA look (Dark Vessel, White BG)"""
        if np.max(img_array) - np.min(img_array) < 1e-6:
            return np.full_like(img_array, 255, dtype=np.uint8)
            
        # Normalize
        img_norm = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
        
        # DSA Contrast Curve (Darken the lows)
        # Power > 1 makes darks darker.
        img_gamma = np.power(img_norm, 3.0) 
        
        return (img_gamma * 255).astype(np.uint8)

    def process_views(self):
        results = []
        
        # --- INIT gVXR WITH TEMP FILE ---
        gvxr.createOpenGLContext()
        gvxr.setSourcePosition(SOURCE_POS_MM[0], SOURCE_POS_MM[1], SOURCE_POS_MM[2], "mm")
        gvxr.usePointSource()
        gvxr.setMonoChromatic(0.06, "MeV", 1000) 
        
        gvxr.setDetectorPosition(DETECTOR_POS_MM[0], DETECTOR_POS_MM[1], DETECTOR_POS_MM[2], "mm")
        gvxr.setDetectorUpVector(0, 0, -1)
        gvxr.setDetectorNumberOfPixels(IMAGE_SIZE, IMAGE_SIZE)
        gvxr.setDetectorPixelSize(PIXEL_SZ_MM, PIXEL_SZ_MM, "mm")

        # Load the ALIGNED temp file
        gvxr.loadMeshFile("vessel", TEMP_MESH_FILE, "mm")
        
        # Material: Iodine
        gvxr.setElement("vessel", "I") 
        gvxr.setDensity("vessel", 1.5, "g/cm3") # Slightly reduced density for better gradients

        for angle in [0, 90]:
            tag = f"ang{angle}"
            
            # 1. Rotate
            if angle != 0:
                gvxr.rotateNode("vessel", angle, 0, 1, 0)
            
            # 2. X-Ray
            raw_xray = np.array(gvxr.computeXRayImage()).astype(np.float32)
            img_uint8 = self.contrast_stretch(raw_xray)
            
            xray_filename = f"{self.base_name}_{tag}.png"
            PIL.Image.fromarray(img_uint8).save(os.path.join(self.output_dir, xray_filename))
            
            # 3. Depth (Ray Trace)
            depth_filename = f"{self.base_name}_{tag}_depth.png"
            self.compute_depth_map(angle, depth_filename)
            
            # 4. Metadata (Projection Matrix)
            f_pix = 1000.0 / PIXEL_SZ_MM # 1000mm throw
            K = [[f_pix, 0, IMAGE_SIZE/2], [0, f_pix, IMAGE_SIZE/2], [0, 0, 1]]
            
            results.append({
                "image_file": xray_filename,
                "depth_file": depth_filename,
                "mesh_source": f"{self.base_name}_mesh",
                "view_angle": angle,
                "projection_matrix": K
            })

            # Reset
            if angle != 0:
                gvxr.rotateNode("vessel", -angle, 0, 1, 0)
        
        # Cleanup Context to free memory
        # (gVXR context is global, so we rely on the loop)
        return results

    def compute_depth_map(self, angle, filename):
        mesh_copy = self.mesh.copy()
        
        # Rotate (Y-Axis)
        rot = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0])
        mesh_copy.apply_transform(rot)
        
        # Camera Grid (Millimeters)
        # Matches gVXR Detector Pixel Size * Count
        half_w = (IMAGE_SIZE * PIXEL_SZ_MM) / 2.0
        y = np.linspace(-half_w, half_w, IMAGE_SIZE)
        z = np.linspace(-half_w, half_w, IMAGE_SIZE)
        yy, zz = np.meshgrid(y, z)
        
        # Pixel Locations (at x = +400)
        pixel_locs = np.stack([np.full_like(yy, DETECTOR_POS_MM[0]), yy, zz], axis=-1).reshape(-1, 3)
        
        # Rays (from x = -600)
        ray_origins = np.tile(SOURCE_POS_MM, (len(pixel_locs), 1))
        ray_dirs = pixel_locs - ray_origins
        ray_dirs /= np.linalg.norm(ray_dirs, axis=1)[:, np.newaxis]
        
        # Cast
        locations, index_ray, _ = mesh_copy.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_dirs,
            multiple_hits=False
        )
        
        # Depth Map Construction
        depth_img = np.zeros(IMAGE_SIZE * IMAGE_SIZE)
        if len(locations) > 0:
            dists = np.linalg.norm(locations - SOURCE_POS_MM, axis=1)
            
            # Dynamic Normalization (Fixes "Ruined" grey images)
            # Instead of fixed 550-650, we use the actual object bounds
            # This ensures maximum contrast in the depth map
            min_d = np.min(dists)
            max_d = np.max(dists)
            if max_d > min_d:
                dists_norm = (dists - min_d) / (max_d - min_d)
                depth_img[index_ray] = dists_norm * 255
            else:
                depth_img[index_ray] = 128 # Flat object
            
        depth_img = depth_img.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)
        PIL.Image.fromarray(depth_img).save(os.path.join(self.output_dir, filename))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    mesh_files = glob.glob(os.path.join(INPUT_DIR, "*_mesh.stl"))
    mesh_files.sort()
    
    print(f"Processing {len(mesh_files)} meshes (Unified MM Pipeline)...")
    
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
            
    # Cleanup temp file
    if os.path.exists(TEMP_MESH_FILE):
        os.remove(TEMP_MESH_FILE)

    with open(os.path.join(OUTPUT_DIR, "dataset.json"), 'w') as f:
        json.dump(dataset_labels, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()