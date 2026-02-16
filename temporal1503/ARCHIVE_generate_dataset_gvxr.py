import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm
import gvxrPython3 as gvxr

# --- CONFIGURATION ---
INPUT_DIR = "output_neuro_final"   # Your sequential/hybrid simulation output
OUTPUT_DIR = "final_dataset"       # Folder for the PINN training data
IMAGE_SIZE = 512

# Medical C-Arm Geometry (in mm)
SOD = 600.0   # Source to Object Distance (Isocenter)
SDD = 1000.0  # Source to Detector Distance
DETECTOR_SIZE_MM = 400.0 
PIXEL_SIZE = DETECTOR_SIZE_MM / IMAGE_SIZE

def compute_projection_matrix(angle_deg):
    """
    Calculates the 3x4 Projection Matrix (P = K * [R|t]) 
    This allows the PINN to understand the relationship between 
    3D space and the 2D pixel coordinates.
    """
    # 1. Intrinsic Matrix (K)
    # Focal length in pixels
    f_pix = SOD / PIXEL_SIZE 
    
    # Principal point (center of image)
    cx = IMAGE_SIZE / 2
    cy = IMAGE_SIZE / 2
    
    K = np.array([
        [f_pix, 0,     cx],
        [0,     f_pix, cy],
        [0,     0,     1 ]
    ])
    
    # 2. Extrinsic Matrix [R|t]
    # We simulate rotating the C-Arm around the patient.
    # Note: Angle is negative because rotating the object +X is 
    # mathematically equivalent to rotating the camera -X.
    angle_rad = np.radians(-angle_deg) 
    
    # Rotation around Y axis (standard orbital rotation)
    R = np.array([
        [np.cos(angle_rad),  0, np.sin(angle_rad)],
        [0,                  1, 0                ],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    
    # Translation: The camera center is at (0, 0, -SOD) in world space
    # t = -R * C
    C = np.array([0, 0, -SOD]) 
    t = -R @ C
    
    # Construct Extrinsic [R | t]
    Extrinsics = np.column_stack((R, t))
    
    # 3. Final Projection P = K @ Extrinsics
    P = K @ Extrinsics
    return P

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 1. INITIALIZE gVXR (The Physics Engine) ---
    print("Initializing gVirtualXRay...")
    
    # Create Context (Headless requires xvfb-run on Linux)
    gvxr.createOpenGLContext()
    
    # Setup Source (X-Ray Tube)
    gvxr.setSourcePosition(0.0, 0.0, -SOD, "mm")
    gvxr.usePointSource()
    
    # PHYSICS FIX: Use 60 keV (0.06 MeV). 
    # This is the "Sweet Spot" for Iodine K-edge absorption.
    # Too high (>100keV) and the dye becomes invisible.
    gvxr.setMonoChromatic(0.06, "MeV", 1000) 
    
    # Setup Detector
    # Detector is placed at (SDD - SOD) relative to isocenter
    det_z = SDD - SOD
    gvxr.setDetectorPosition(0.0, 0.0, det_z, "mm")
    gvxr.setDetectorUpVector(0, -1, 0) # Standard medical orientation
    gvxr.setDetectorNumberOfPixels(IMAGE_SIZE, IMAGE_SIZE)
    gvxr.setDetectorPixelSize(PIXEL_SIZE, PIXEL_SIZE, "mm")
    
    # --- 2. PROCESS LOOP ---
    mesh_files = glob.glob(os.path.join(INPUT_DIR, "*_mesh.stl"))
    mesh_files.sort()
    
    if len(mesh_files) == 0:
        print(f"CRITICAL ERROR: No .stl files found in {INPUT_DIR}")
        return

    print(f"Found {len(mesh_files)} meshes. Simulating Angiograms...")
    
    dataset_labels = {}
    
    for mesh_path in tqdm(mesh_files):
        try:
            # Base name: branch_00_pass_00_frame_000
            base_name = os.path.basename(mesh_path).replace("_mesh.stl", "")
            
            # 1. Load Mesh
            # We assume the vessel is filled with contrast dye.
            gvxr.loadMeshFile("vessel", mesh_path, "mm")
            
            # --- MATERIAL SETTINGS (Angiogram) ---
            # Use Iodine (Z=53), the active element in contrast media
            gvxr.setElement("vessel", "I") 
            
            # Set Density to ~1.35 g/cm3 (Typical for liquid Iohexol/Omnipaque)
            gvxr.setDensity("vessel", 1.35, "g/cm3") 
            
            gvxr.moveToCentre("vessel") # Ensure it's at isocenter (0,0,0)
            
            # 2. Generate Views (AP: 0 deg, Lateral: 90 deg)
            for angle in [0, 90]:
                
                # Apply Rotation
                if angle != 0:
                    gvxr.rotateNode("vessel", angle, 0, 1, 0)
                
                # 3. Compute X-Ray (Ray Tracing + Attenuation Law)
                # Returns total photon energy absorbed
                raw_xray = np.array(gvxr.computeXRayImage()).astype(np.float32)
                
                # 4. Post-Processing (Simulate Fluoroscopy Look)
                # Invert: High attenuation (Iodine) should be DARK. Background BRIGHT.
                
                # Normalize 0 to 1 based on min/max in scene
                if np.max(raw_xray) > np.min(raw_xray):
                    img_norm = (raw_xray - np.min(raw_xray)) / (np.max(raw_xray) - np.min(raw_xray))
                else:
                    img_norm = np.zeros_like(raw_xray)
                
                # Invert (X-ray negative)
                img_inverted = 1.0 - img_norm
                
                # Convert to 8-bit image
                img_final = (img_inverted * 255).astype(np.uint8)
                
                # Save Image
                img_filename = f"{base_name}_ang{angle}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, img_filename), img_final)
                
                # Calculate Projection Matrix for this specific view
                proj_mat = compute_projection_matrix(angle)
                
                # Add to JSON
                dataset_labels[img_filename] = {
                    "mesh_source": f"{base_name}_mesh",
                    "view_angle": angle,
                    "projection_matrix": proj_mat.tolist()
                }
                
                # Reset Rotation for next loop iteration
                if angle != 0:
                    gvxr.rotateNode("vessel", -angle, 0, 1, 0)
            
            # Clean up mesh from memory for next iteration
            gvxr.removePolygonMeshes("vessel")

        except Exception as e:
            print(f"Error processing {mesh_path}: {e}")
            # Try to cleanup if crash happened
            try: gvxr.removePolygonMeshes("vessel")
            except: pass

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_labels, f, indent=4)
        
    print(f"\nDone! Generated {len(dataset_labels)} realistic Angiograms in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main() 