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
INPUT_DIR = "output_sequential_wt"
OUTPUT_DIR = "output_gvxr_6"
TEMP_MESH_FILE = "temp_geometry.stl" 
IMAGE_SIZE = 4096

# Geometry (Millimeters)
SOURCE_POS_MM   = np.array([-600.0, 0.0, 0.0]) 
DETECTOR_POS_MM = np.array([ 400.0, 0.0, 0.0]) 
DETECTOR_SIZE_MM = 300.0 
PIXEL_SZ_MM = DETECTOR_SIZE_MM / IMAGE_SIZE

class XRayDepthGenerator:
    def __init__(self, mesh_path, output_dir):
        self.original_path = mesh_path
        self.output_dir = output_dir
        self.base_name = os.path.basename(mesh_path).replace("_mesh.stl", "")
        
        # Load and Standardize Mesh
        self.mesh = trimesh.load(self.original_path, process=False)
        box_center = self.mesh.bounding_box.centroid
        self.mesh.apply_translation(-box_center)
        
        # Unit Fix: Ensure size is substantial (>50mm)
        if np.max(self.mesh.bounding_box.extents) < 50.0: 
            self.mesh.apply_scale(10.0) 
            
        self.mesh.export(TEMP_MESH_FILE)

    def adaptive_contrast_stretch(self, img_array):
        """
        Smart contrast: Finds the vessel signal and stretches it to fill 0-255.
        """
        # 1. Background Estimation (Mode of histogram is likely background)
        bg_val = np.percentile(img_array, 95) 
        
        # 2. Vessel Signal (Darkest part)
        vessel_val = np.min(img_array)
        
        # Check if there is actual signal (avoid amplifying noise)
        if (bg_val - vessel_val) < 1e-6:
            return np.full(img_array.shape, 255, dtype=np.uint8) 

        # 3. Clip & Stretch
        img_clipped = np.clip(img_array, vessel_val, bg_val)
        img_norm = (img_clipped - vessel_val) / (bg_val - vessel_val)
        
        # 4. Gamma Boost (Make mid-tones darker)
        img_gamma = np.power(img_norm, 2.0)
        
        # Scale to 0-255 (0=Black Vessel, 255=White BG)
        return (img_gamma * 255).astype(np.uint8)

    def process_views(self):
        results = []
        
        # Setup gVXR
        gvxr.createOpenGLContext()
        gvxr.setSourcePosition(SOURCE_POS_MM[0], SOURCE_POS_MM[1], SOURCE_POS_MM[2], "mm")
        gvxr.usePointSource()
        gvxr.setMonoChromatic(0.06, "MeV", 1000) 
        
        gvxr.setDetectorPosition(DETECTOR_POS_MM[0], DETECTOR_POS_MM[1], DETECTOR_POS_MM[2], "mm")
        gvxr.setDetectorUpVector(0, 0, -1)
        gvxr.setDetectorNumberOfPixels(IMAGE_SIZE, IMAGE_SIZE)
        gvxr.setDetectorPixelSize(PIXEL_SZ_MM, PIXEL_SZ_MM, "mm")

        gvxr.loadMeshFile("vessel", TEMP_MESH_FILE, "mm")
        
        # Material: High Density Iodine
        gvxr.setElement("vessel", "I") 
        gvxr.setDensity("vessel", 2.5, "g/cm3") 

        # --- VIEW DEFINITIONS ---
        # Format: (Name, LAO/RAO angle, Cranial/Caudal angle)
        # LAO = Positive Y rotation, RAO = Negative Y rotation
        # Cranial = Positive Z rotation, Caudal = Negative Z rotation (Approx)
        views = [
            ("AP", 0, 0),
            ("Lateral", 90, 0),
            ("LAO45", 45, 0),
            ("RAO45", -45, 0),
            ("Spider", 45, 20),          # LAO 45 + Cranial 20
            ("RAO30_Caudal20", -30, -20) # RAO 30 + Caudal 20
        ]

        for view_name, lao_angle, cran_angle in views:
            
            # Apply Rotations
            # 1. LAO/RAO (Rotation around Y axis)
            if lao_angle != 0:
                gvxr.rotateNode("vessel", lao_angle, 0, 1, 0)
            
            # 2. Cranial/Caudal (Rotation around Z axis - perpendicular to beam X and patient Y)
            if cran_angle != 0:
                gvxr.rotateNode("vessel", cran_angle, 0, 0, 1)
            
            # X-Ray Generation
            raw_xray = np.array(gvxr.computeXRayImage()).astype(np.float32)
            
            # Contrast Logic
            img_uint8 = self.adaptive_contrast_stretch(raw_xray)
            
            xray_filename = f"{self.base_name}_{view_name}.png"
            PIL.Image.fromarray(img_uint8).save(os.path.join(self.output_dir, xray_filename))
            
            # Depth Generation
            depth_filename = f"{self.base_name}_{view_name}_depth.png"
            self.compute_depth_map(lao_angle, cran_angle, depth_filename)
            
            # Metadata
            f_pix = 1000.0 / PIXEL_SZ_MM 
            K = [[f_pix, 0, IMAGE_SIZE/2], [0, f_pix, IMAGE_SIZE/2], [0, 0, 1]]
            
            results.append({
                "image_file": xray_filename,
                "depth_file": depth_filename,
                "mesh_source": f"{self.base_name}_mesh",
                "view_name": view_name,
                "angles_deg": [lao_angle, cran_angle],
                "projection_matrix": K
            })

            # RESET Rotations (Apply inverse in reverse order)
            if cran_angle != 0:
                gvxr.rotateNode("vessel", -cran_angle, 0, 0, 1)
            if lao_angle != 0:
                gvxr.rotateNode("vessel", -lao_angle, 0, 1, 0)
        
        return results

    def compute_depth_map(self, lao_angle, cran_angle, filename):
        mesh_copy = self.mesh.copy()
        
        # Apply Compound Rotation to match gVXR
        # gVXR applied Y first, then Z. Trimesh applies transforms sequentially.
        
        # 1. LAO/RAO (Y-axis)
        rot_y = trimesh.transformations.rotation_matrix(np.radians(lao_angle), [0, 1, 0])
        mesh_copy.apply_transform(rot_y)
        
        # 2. Cranial/Caudal (Z-axis)
        rot_z = trimesh.transformations.rotation_matrix(np.radians(cran_angle), [0, 0, 1])
        mesh_copy.apply_transform(rot_z)
        
        half_w = (IMAGE_SIZE * PIXEL_SZ_MM) / 2.0
        y = np.linspace(-half_w, half_w, IMAGE_SIZE)
        z = np.linspace(-half_w, half_w, IMAGE_SIZE)
        yy, zz = np.meshgrid(y, z)
        
        pixel_locs = np.stack([np.full_like(yy, DETECTOR_POS_MM[0]), yy, zz], axis=-1).reshape(-1, 3)
        ray_origins = np.tile(SOURCE_POS_MM, (len(pixel_locs), 1))
        ray_dirs = pixel_locs - ray_origins
        ray_dirs /= np.linalg.norm(ray_dirs, axis=1)[:, np.newaxis]
        
        locations, index_ray, _ = mesh_copy.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_dirs, multiple_hits=False
        )
        
        depth_img = np.zeros(IMAGE_SIZE * IMAGE_SIZE)
        if len(locations) > 0:
            dists = np.linalg.norm(locations - SOURCE_POS_MM, axis=1)
            # Auto-Scale Depth to maximize visibility
            min_d, max_d = np.min(dists), np.max(dists)
            if max_d > min_d:
                dists_norm = (dists - min_d) / (max_d - min_d)
                depth_img[index_ray] = dists_norm * 255
            else:
                depth_img[index_ray] = 200 
            
        depth_img = depth_img.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)
        PIL.Image.fromarray(depth_img).save(os.path.join(self.output_dir, filename))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    mesh_files = glob.glob(os.path.join(INPUT_DIR, "*_mesh.stl"))
    mesh_files.sort()
    
    print(f"Processing {len(mesh_files)} meshes (6 Clinical Views)...")
    
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
            import traceback
            traceback.print_exc()
            
    if os.path.exists(TEMP_MESH_FILE):
        os.remove(TEMP_MESH_FILE)

    with open(os.path.join(OUTPUT_DIR, "dataset.json"), 'w') as f:
        json.dump(dataset_labels, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    main()