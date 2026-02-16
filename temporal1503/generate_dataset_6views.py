import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm

# --- IMPORT LOGIC ---
try:
    from gvxrPython3 import gvxr
except ImportError:
    import gvxr

# --- CONFIGURATION ---
INPUT_DIR = "output_sequential_second"
OUTPUT_DIR = "output_gvxr"
IMAGE_SIZE = 512
VIEWS_TO_GENERATE = [0, 90, -90, 45, -45, 30] 

# Medical C-Arm Geometry (in mm)
SOD = 600.0   
SDD = 1000.0  
DETECTOR_SIZE_MM = 400.0 
PIXEL_SIZE = DETECTOR_SIZE_MM / IMAGE_SIZE

def compute_projection_matrix(angle_deg):
    f_pix = SOD / PIXEL_SIZE 
    cx = IMAGE_SIZE / 2
    cy = IMAGE_SIZE / 2
    K = np.array([[f_pix, 0, cx], [0, f_pix, cy], [0, 0, 1]])
    
    angle_rad = np.radians(-angle_deg) 
    R = np.array([
        [np.cos(angle_rad),  0, np.sin(angle_rad)],
        [0,                  1, 0                ],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    C = np.array([0, 0, -SOD]) 
    t = -R @ C
    Extrinsics = np.column_stack((R, t))
    return K @ Extrinsics

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Initializing gVirtualXRay...")
    
    # Create Window (Try-Catch for different backends)
    try:
        gvxr.createWindow(0, IMAGE_SIZE, IMAGE_SIZE, "OPENGL")
    except:
        gvxr.createWindow()
        
    gvxr.setWindowSize(IMAGE_SIZE, IMAGE_SIZE)
    
    # Setup Source/Detector
    gvxr.setSourcePosition(0.0, 0.0, -SOD, "mm")
    gvxr.usePointSource()
    gvxr.setMonoChromatic(0.06, "MeV", 1000) 
    
    det_z = SDD - SOD
    gvxr.setDetectorPosition(0.0, 0.0, det_z, "mm")
    gvxr.setDetectorUpVector(0, -1, 0)
    gvxr.setDetectorNumberOfPixels(IMAGE_SIZE, IMAGE_SIZE)
    gvxr.setDetectorPixelSize(PIXEL_SIZE, PIXEL_SIZE, "mm")
    
    mesh_files = glob.glob(os.path.join(INPUT_DIR, "*_mesh.stl"))
    mesh_files.sort()
    
    if len(mesh_files) == 0:
        print(f"CRITICAL ERROR: No .stl files found in {INPUT_DIR}")
        return

    print(f"Found {len(mesh_files)} meshes. Simulating {len(VIEWS_TO_GENERATE)} views per mesh...")
    
    dataset_labels = {}
    
    for mesh_path in tqdm(mesh_files):
        try:
            # SAFETY CLEAR: Ensure scene is empty before loading
            # This prevents "Node already exists" errors
            gvxr.emptyScene() 

            base_name = os.path.basename(mesh_path).replace("_mesh.stl", "")
            
            # Load Mesh
            gvxr.loadMeshFile("vessel", mesh_path, "mm")
            gvxr.setElement("vessel", "I") 
            gvxr.setDensity("vessel", 1.35, "g/cm3") 
            gvxr.moveToCentre("vessel") 
            
            for angle in VIEWS_TO_GENERATE:
                gvxr.rotateNode("vessel", angle, 0, 1, 0)
                
                # Render
                raw_xray = np.array(gvxr.computeXRayImage()).astype(np.float32)
                
                # Check for white screen (all zeros or all ones)
                if np.sum(raw_xray) == 0:
                    # If image is empty, skip saving or it will mess up training
                    # Reset rotation and continue
                    gvxr.rotateNode("vessel", -angle, 0, 1, 0)
                    continue

                # Normalize & Invert
                min_val = np.min(raw_xray)
                max_val = np.max(raw_xray)
                
                if max_val > min_val:
                    img_norm = (raw_xray - min_val) / (max_val - min_val)
                else:
                    img_norm = np.zeros_like(raw_xray)
                
                img_final = ((1.0 - img_norm) * 255).astype(np.uint8)
                
                # Save
                angle_str = f"{angle}" if angle >= 0 else f"neg{abs(angle)}"
                img_filename = f"{base_name}_ang{angle_str}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, img_filename), img_final)
                
                proj_mat = compute_projection_matrix(angle)
                dataset_labels[img_filename] = {
                    "mesh_source": f"{base_name}_mesh",
                    "view_angle": angle,
                    "projection_matrix": proj_mat.tolist()
                }
                
                # Reset Rotation
                gvxr.rotateNode("vessel", -angle, 0, 1, 0)
            
            # Clean up (The correct API call)
            gvxr.removeNode("vessel")

        except Exception as e:
            print(f"Error processing {mesh_path}: {e}")
            # Force scene clear if crash happens
            try: gvxr.emptyScene()
            except: pass

    with open(os.path.join(OUTPUT_DIR, "dataset.json"), 'w') as f:
        json.dump(dataset_labels, f, indent=4)
        
    print(f"\nDone! Dataset saved to '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
