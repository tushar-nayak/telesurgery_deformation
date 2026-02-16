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
INPUT_DIR = "output_sequential_second" # Ensure this matches your actual data folder name
OUTPUT_DIR = "output_gvxr_3"
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
        # Field of View: 300mm x 300mm (Matches a typical neuro head view)
        gvxr.setDetectorPixelSize(300.0/IMAGE_SIZE, 300.0/IMAGE_SIZE, "mm")

        # --- KEY FIX 1: UNIT SCALING ---
        # We assume the STL is in CM. We load it as "cm" so gVXR treats '1.0' as '10mm'.
        gvxr.loadMeshFile("vessel", self.mesh_path, "cm")
        
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
            
            # --- KEY FIX 2: CONTRAST BOOST ---
            # Power > 1.0 makes dark things (vessel) darker.
            # Power < 1.0 makes dark things lighter (washed out).
            # We want the vessel to be distinct black.
            img_enhanced = np.power(img_norm, 4.0) 
            
            img_uint8 = (img_enhanced * 255).astype(np.uint8)

            # Save X-Ray
            xray_filename = f"{self.base_name}_{tag}.png"
            xray_path = os.path.join(self.output_dir, xray_filename)
            PIL.Image.fromarray(img_uint8).save(xray_path)
            
            # 3. Compute Depth
            depth_filename = f"{self.base_name}_{tag}_depth.png"
            self.compute_depth_map(angle, depth_filename)
            
            # 4. Save Metadata
            # Note: We update P matrix focal length to match the new Pixel Size (300mm FOV)
            f_pix = 1000.0 / (300.0/IMAGE_SIZE) # SDD / PixelSize
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
        mesh_copy = self.mesh.copy()
        
        # Apply Rotation manually to Trimesh
        rot = trimesh.transformations.rotation_matrix(np.radians(angle), [0, 1, 0])
        mesh_copy.apply_transform(rot)
        
        # Source/Detector positions (Trimesh units must match loaded units)
        # We loaded 'cm' into GVXR, so GVXR scaled it up x10 internally.
        # Here we just use consistent units. Let's stick to the STL's native units 
        # and scale the camera accordingly to match the visual ratio.
        # If STL is ~20 units, and we want it to look like the X-ray (which is scaled to 200mm),
        # we need to ensure the geometry ratio matches.
        
        # Easier way: Source is -600mm, Detector +400mm. Total = 1000mm.
        # Detector Width = 300mm.
        # If STL is in CM, Source = -60.0, Detector = +40.0. Width = 30.0.
        
        source_pos = np.array([-60.0, 0, 0])
        
        # Create Grid
        y = np.linspace(-15.0, 15.0, IMAGE_SIZE) # 30cm width
        z = np.linspace(-15.0, 15.0, IMAGE_SIZE)
        yy, zz = np.meshgrid(y, z)
        
        pixel_locs = np.stack([np.full_like(yy, 40.0), yy, zz], axis=-1).reshape(-1, 3)
        
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
            # Normalize 50-70cm range
            dists_norm = (dists - 50.0) / (70.0 - 50.0)
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
    
    print(f"Processing {len(mesh_files)} meshes (DSA Style v2)...")
    
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