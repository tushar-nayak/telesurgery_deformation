import numpy as np
import trimesh
import matplotlib.pyplot as plt
import os
from glob import glob
import scipy.ndimage
import json

# --- 1. C-Arm Geometry Math ---
def create_projection_matrix(lao_rao_angle, cran_caud_angle):
    """
    Creates a rotation matrix to simulate C-arm angles.
    """
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

# --- 2. Medical Realism (The DSA Look) ---
def apply_dsa_style(image_array, blur_radius=1.2, noise_level=0.008):
    # Gaussian Blur
    blurred = scipy.ndimage.gaussian_filter(image_array, sigma=blur_radius)
    
    # Normalize
    if blurred.max() > 0:
        img_norm = (blurred - blurred.min()) / (blurred.max() - blurred.min())
    else:
        img_norm = blurred

    # Poisson Noise
    noise = np.random.poisson(img_norm / noise_level) * noise_level
    noisy_image = img_norm + (noise - img_norm) * 0.7
    
    # Contrast Stretching
    contrast_image = np.clip(noisy_image, 0.1, 0.9)
    contrast_image = (contrast_image - 0.1) / 0.8
    
    return np.clip(contrast_image, 0, 1)

# --- 3. The X-Ray Generator ---
def generate_synthetic_dsa(mesh, projection_matrix, image_size=512, pitch=0.5):
    # Rotate & Voxelize
    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(projection_matrix)
    voxel_grid = mesh_copy.voxelized(pitch=pitch).fill()
    
    # Project & Invert (Beer-Lambert)
    density_map = np.sum(voxel_grid.matrix, axis=2)
    mu = 0.6  
    attenuation = np.exp(-mu * density_map)
    dsa_raw = 1.0 - attenuation
    
    # Apply Style
    final_image = apply_dsa_style(dsa_raw)
    
    # Center & Pad
    h, w = final_image.shape
    pad_h = max(0, image_size - h)
    pad_w = max(0, image_size - w)
    padded = np.pad(final_image, 
                   ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)),
                   mode='constant', constant_values=0)
    
    start_h = (padded.shape[0] - image_size) // 2
    start_w = (padded.shape[1] - image_size) // 2
    return padded[start_h:start_h+image_size, start_w:start_w+image_size]

# --- Main Execution ---
if __name__ == "__main__":
    input_dir = "output_meshes"
    output_dir = "output_dsa_images"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    mesh_files = glob(os.path.join(input_dir, "*.stl"))
    
    # Dictionary to store labels for the Neural Network
    dataset_labels = {} 

    if not mesh_files:
        print(f"No meshes found in '{input_dir}'.")
    else:
        print(f"Found {len(mesh_files)} meshes. Generating DSA images + Labels...")
        
        views = [
            {"name": "AP", "lao": 0, "cran": 0},          
            {"name": "LAT", "lao": 90, "cran": 0},        
            {"name": "LAO30", "lao": 30, "cran": 0},      
            {"name": "RAO30_CRAN15", "lao": -30, "cran": 15} 
        ]

        for i, mesh_path in enumerate(mesh_files):
            mesh_name = os.path.basename(mesh_path).replace(".stl", "")
            print(f"[{i+1}/{len(mesh_files)}] Processing {mesh_name}...")
            
            try:
                mesh = trimesh.load_mesh(mesh_path)
                
                for view in views:
                    # 1. Calc Matrix
                    proj_mat = create_projection_matrix(view['lao'], view['cran'])
                    
                    # 2. Generate DSA
                    dsa_img = generate_synthetic_dsa(mesh, proj_mat, image_size=512, pitch=0.5)
                    
                    # 3. Save Image
                    image_filename = f"{mesh_name}_{view['name']}.png"
                    save_path = os.path.join(output_dir, image_filename)
                    plt.imsave(save_path, dsa_img, cmap='gray', vmin=0, vmax=1)
                    
                    # 4. SAVE LABELS (The crucial part)
                    # We store the inputs needed for your Neural Network
                    dataset_labels[image_filename] = {
                        "mesh_source": mesh_name,
                        "view_name": view['name'],
                        "camera_angles": {
                            "lao": view['lao'],
                            "cran": view['cran']
                        },
                        # Save the 4x4 matrix as a list so JSON can read it
                        "projection_matrix": proj_mat.tolist() 
                    }
                    
            except Exception as e:
                print(f"  Error processing {mesh_name}: {e}")

        # 5. Write the Label File
        json_path = os.path.join(output_dir, "dataset_labels.json")
        with open(json_path, 'w') as f:
            json.dump(dataset_labels, f, indent=4)
            
        print(f"\nSuccess! Images and 'dataset_labels.json' saved to {output_dir}")