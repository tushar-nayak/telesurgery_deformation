import numpy as np
import trimesh
import matplotlib.pyplot as plt
import os
from glob import glob
import scipy.ndimage
import json

# --- 1. C-Arm Geometry Math ---
def create_projection_matrix(lao_rao_angle, cran_caud_angle):
    alpha = np.radians(lao_rao_angle) 
    beta = np.radians(cran_caud_angle)

    # Rotation around Y (LAO/RAO)
    R_y = np.array([
        [np.cos(alpha), 0, np.sin(alpha), 0],
        [0, 1, 0, 0],
        [-np.sin(alpha), 0, np.cos(alpha), 0],
        [0, 0, 0, 1]
    ])

    # Rotation around X (CRAN/CAUD)
    R_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(beta), -np.sin(beta), 0],
        [0, np.sin(beta), np.cos(beta), 0],
        [0, 0, 0, 1]
    ])

    return np.dot(R_x, R_y)

# --- 2. Fluoroscopy Style (HD Tuned) ---
def apply_fluoroscopy_style(image_array, blur_radius=2.0, noise_level=0.003):
    """
    HD Update:
    - Reduced noise_level (0.005 -> 0.003) for cleaner look
    - Increased blur_radius (1.2 -> 2.0) to match higher pixel density
    """
    # 1. Blur
    blurred = scipy.ndimage.gaussian_filter(image_array, sigma=blur_radius)
    
    # 2. Add Noise (Photon Shot Noise)
    noise = np.random.poisson(blurred / noise_level) * noise_level
    
    # Blend noise (80% original, 20% noisy) - Cleaner than before
    noisy_image = blurred + (noise - blurred) * 0.2
    
    # 3. Contrast Stretch
    contrast_image = np.clip(noisy_image, 0, 1)
    contrast_image = contrast_image ** 1.2 # Gamma correction
    
    return np.clip(contrast_image, 0, 1)

# --- 3. The X-Ray Generator (High Res) ---
def generate_synthetic_xray(mesh, projection_matrix, image_size=1024, pitch=0.2):
    """
    HD Update:
    - Default image_size increased to 1024
    - Default pitch reduced to 0.2 (finer voxels)
    """
    # Rotate
    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(projection_matrix)
    
    # Voxelize (High Res)
    # WARNING: pitch=0.2 is 15x slower than 0.5, but looks much better.
    # If it is too slow, try 0.3
    voxel_grid = mesh_copy.voxelized(pitch=pitch).fill()
    
    # Project
    density_map = np.sum(voxel_grid.matrix, axis=2)
    
    # Beer-Lambert Law
    mu = 0.6  
    attenuation = np.exp(-mu * density_map)
    
    # Black on White Logic
    raw_xray = attenuation
    
    # Apply Style
    final_image = apply_fluoroscopy_style(raw_xray)
    
    # Center & Pad
    h, w = final_image.shape
    pad_h = max(0, image_size - h)
    pad_w = max(0, image_size - w)
    
    padded = np.pad(final_image, 
                   ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)),
                   mode='constant', constant_values=1.0)
    
    start_h = (padded.shape[0] - image_size) // 2
    start_w = (padded.shape[1] - image_size) // 2
    return padded[start_h:start_h+image_size, start_w:start_w+image_size]

# --- Main Execution ---
if __name__ == "__main__":
    input_dir = "output_meshes"
    output_dir = "output_fluoroscopy_highres" 
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    mesh_files = glob(os.path.join(input_dir, "*.stl"))
    dataset_labels = {} 

    if not mesh_files:
        print(f"No meshes found in '{input_dir}'.")
    else:
        print(f"Found {len(mesh_files)} meshes. Generating High-Res images...")
        print("Note: This will take longer due to pitch=0.2 voxelization.")

        for i, mesh_path in enumerate(mesh_files):
            mesh_name = os.path.basename(mesh_path).replace(".stl", "")
            print(f"[{i+1}/{len(mesh_files)}] Processing {mesh_name}...")
            
            try:
                mesh = trimesh.load_mesh(mesh_path)
                
                # Generate 10 Random Views
                for view_idx in range(10):
                    lao = np.random.randint(-90, 91)
                    cran = np.random.randint(-45, 46)
                    view_name = f"view_{view_idx}_LAO{lao}_CRAN{cran}"
                    
                    proj_mat = create_projection_matrix(lao, cran)
                    
                    # --- CALLING WITH HD PARAMETERS ---
                    xray_img = generate_synthetic_xray(
                        mesh, 
                        proj_mat, 
                        image_size=1024, # 2x Resolution
                        pitch=0.2        # 2.5x finer detail
                    )
                    
                    # Save
                    image_filename = f"{mesh_name}_{view_name}.png"
                    save_path = os.path.join(output_dir, image_filename)
                    plt.imsave(save_path, xray_img, cmap='gray', vmin=0, vmax=1)
                    
                    dataset_labels[image_filename] = {
                        "mesh_source": mesh_name,
                        "view_index": view_idx,
                        "camera_angles": {"lao": int(lao), "cran": int(cran)},
                        "projection_matrix": proj_mat.tolist() 
                    }
                    
            except Exception as e:
                print(f"  Error processing {mesh_name}: {e}")

        with open(os.path.join(output_dir, "dataset_labels.json"), 'w') as f:
            json.dump(dataset_labels, f, indent=4)
            
        print(f"\nSuccess! High-Res images saved to '{output_dir}'")