import os
import glob
import json
import numpy as np
import PIL.Image
import trimesh
import matplotlib
matplotlib.use('Agg') # Headless mode
import matplotlib.pyplot as plt
from tqdm import tqdm
from gvxrPython3 import gvxr

# --- CONFIGURATION ---
INPUT_DIR = "output_neuro_final"
OUTPUT_DIR = "final_dataset"
IMAGE_SIZE = 512

class XRayDepthGenerator:
    def __init__(self, mesh_path, output_dir):
        self.mesh_path = mesh_path
        self.output_dir = output_dir
        self.base_name = os.path.basename(mesh_path).replace("_mesh.stl", "")
        
        # Load mesh for Depth Calculation (Trimesh)
        self.mesh = trimesh.load(self.mesh_path, process=False)
        self.centroid = self.mesh.bounding_box.centroid
        self.mesh.apply_translation(-self.centroid) # Center it

    def setup_gvxr(self):
        gvxr.createOpenGLContext()
        gvxr.setSourcePosition(-600.0, 0.0, 0.0, "mm") 
        gvxr.usePointSource()
        
        # 60 keV is standard for Iodine contrast visibility
        gvxr.setMonoChromatic(0.06, "MeV", 1000) 
        
        gvxr.setDetectorPosition(400.0, 0.0, 0.0, "mm")
        gvxr.setDetectorUpVector(0, 0, -1)
        gvxr.setDetectorNumberOfPixels(IMAGE_SIZE, IMAGE_SIZE)
        gvxr.setDetectorPixelSize(0.5, 0.5, "mm")

        # Load mesh
        gvxr.loadMeshFile("vessel", self.mesh_path, "mm")
        gvxr.moveToCentre("vessel")
        
        # --- MATERIAL: Iodine Contrast Dye ---
        gvxr.setElement("vessel", "I") 
        gvxr.setDensity("vessel", 2.0, "g/cm3") 

    def process_views(self):
        results = []
        self.setup_gvxr()
        
        for angle in [0, 90]:
            tag = f"ang{angle}"
            
            # 1. Rotate GVXR Node
            if angle != 0:
                gvxr.rotateNode("vessel", angle, 0, 1, 0)
                
            # 2. Compute X-Ray (Detector Image)
            # High values = Background (White), Low values = Vessel (Dark)
            raw_xray = np.array(gvxr.computeXRayImage()).astype(np.float32)
            
            # Normalize to 0-1
            if np.max(raw_xray) - np.min(raw_xray) < 1e-6:
                img_norm = np.ones_like(raw_xray) # All white if invisible
            else:
                img_norm = (raw_xray - np.min(raw_xray)) / (np.max(raw_xray) - np.min(raw_xray))
            
            # --- DSA LOOK (White Background, Black Vessel) ---
            # We do NOT invert here. We keep 1.0 as White.
            # Optional: Apply Gamma/Power to darken the vessel further
            img_enhanced = np.power(img_norm, 0.5) # Gamma correction for contrast
            
            img_uint8 = (img_enhanced * 255).astype(np.uint8)

            # Save X-Ray
            xray_filename = f"{self.base_name}_{tag}.png"
            xray_path = os.path.join(self.output_dir, xray_filename)
            PIL.Image.fromarray(img_uint8).save(xray_path)
            
            # 3. Compute Depth
            depth_filename = f"{self.base_name}_{tag}_depth.png"
            self.compute_depth_map(angle, depth_filename)
            
            # 4. Save Metadata
            results.append({
                "image_file": xray_filename,
                "depth_file": depth_filename,
                "mesh_source": f"{self.base_name}_mesh",
                "view_angle": angle,
                "projection_matrix": [[1000, 0, 256, 0], [0, 1000, 256, 0], [0, 0, 1, 0]]
            })

            # Reset Rotation
            if angle != 0:
                gvxr.rotateNode("vessel", -angle, 0, 1, 0)
                
        return results

    def compute_depth_map(self, angle, filename):
        mesh_copy = self.mesh.copy()
        
        # Apply Rotation manually to Trimesh
        rot = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0])
        mesh_copy.apply_transform(rot)
        
        # Source/Detector positions
        source_pos = np.array([-600.0, 0, 0])
        
        # Create Grid
        y = np.linspace(-128, 128, IMAGE_SIZE)
        z = np.linspace(-128, 128, IMAGE_SIZE)
        yy, zz = np.meshgrid(y, z)
        
        pixel_locs = np.stack([np.full_like(yy, 400.0), yy, zz], axis=-1).reshape(-1, 3)
        
        # Ray Directions
        ray_origins = np.tile(source_pos, (len(pixel_locs), 1))
        ray_dirs = pixel_locs - ray_origins
        ray_dirs /= np.linalg.norm(ray_dirs, axis=1)[:, np.newaxis]
        
        # Intersect
        locations, index_ray, _ = mesh_copy.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_dirs,
            multiple_hits=False
        )
        
        # Map to image
        depth_img = np.zeros(IMAGE_SIZE * IMAGE_SIZE)
        if len(locations) > 0:
            dists = np.linalg.norm(locations - source_pos, axis=1)
            # Normalize 500-700mm
            dists_norm = (dists - 550) / (650 - 550)
            dists_norm = np.clip(dists_norm, 0, 1)
            depth_img[index_ray] = dists_norm * 255
            
        depth_img = depth_img.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)
        
        cv2_path = os.path.join(self.output_dir, filename)
        PIL.Image.fromarray(depth_img).save(cv2_path)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    mesh_files = glob.glob(os.path.join(INPUT_DIR, "*_mesh.stl"))
    mesh_files.sort()
    
    print(f"Processing {len(mesh_files)} meshes (DSA Style)...")
    
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
            
    # Save JSON
    with open(os.path.join(OUTPUT_DIR, "dataset.json"), 'w') as f:
        json.dump(dataset_labels, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()