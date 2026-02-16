import numpy as np
import trimesh
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import copy

# --- 1. Load STL & Extract Centerline ---
def load_and_skeletonize(stl_path, voxel_pitch=0.5):
    """
    Standard extraction. Note: For complex branching vessels, 
    the list order might jump between branches. This is fine for 
    deformation physics (which is spatial) but looks bad if plotted as a line.
    """
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"Could not find file: {stl_path}")

    print(f"Loading {stl_path}...")
    mesh = trimesh.load_mesh(stl_path)
    
    # Use a slightly coarser pitch (0.5 or 0.8) to keep point count manageable
    print("Voxelizing (this may take a moment)...")
    voxel_grid = mesh.voxelized(pitch=voxel_pitch).fill()
    
    print("Skeletonizing...")
    skeleton = skeletonize(voxel_grid.matrix)
    indices = np.argwhere(skeleton)
    centerline_points = voxel_grid.indices_to_points(indices)
    
    return mesh, centerline_points

# --- 2. LOCALIZED Deformation Physics (Gaussian) ---
def gaussian_kernel(r, sigma=5.0):
    """
    Gaussian decays to zero quickly. 
    Points > 3*sigma away will effectively not move.
    """
    return np.exp(-r**2 / (2 * sigma**2))

def apply_single_localized_deformation(original_mesh, original_centerline, 
                                       force_scale=15.0, # HIGH magnitude to be visible
                                       sigma=4.0):       # Radius of the "bump" in mm
    """
    Creates ONE copy of the vessel with ONE localized deformation.
    """
    # Create deep copies so we don't destroy the original
    mesh = original_mesh.copy()
    centerline = original_centerline.copy()
    
    # 1. Pick ONE random "Epicenter" (The poke site)
    center_idx = np.random.randint(0, len(centerline))
    epicenter = centerline[center_idx]
    
    # 2. Generate ONE random force vector
    force = np.random.uniform(-1, 1, size=3)
    # Normalize and scale
    force = (force / np.linalg.norm(force)) * force_scale
    
    # --- A. Deform Mesh (Gaussian Weighting) ---
    # Calculate spatial distance from every vertex to the epicenter
    dists_mesh = np.linalg.norm(mesh.vertices - epicenter, axis=1)
    
    # Calculate weights (1.0 at epicenter, 0.0 far away)
    weights_mesh = gaussian_kernel(dists_mesh, sigma=sigma)
    
    # Apply displacement: weight * force
    # reshaping to (N, 3) to apply vector to all points
    disp_mesh = np.outer(weights_mesh, force)
    mesh.vertices += disp_mesh
    
    # --- B. Deform Centerline ---
    dists_cl = np.linalg.norm(centerline - epicenter, axis=1)
    weights_cl = gaussian_kernel(dists_cl, sigma=sigma)
    disp_cl = np.outer(weights_cl, force)
    centerline += disp_cl
    
    # Metadata to record "Where" and "How much"
    metadata = {
        "epicenter_index": center_idx,
        "epicenter_xyz": epicenter,
        "applied_force": force
    }
    
    return mesh, centerline, metadata

# --- Main Execution ---
if __name__ == "__main__":
    stl_filename = "vessel.stl"  # CHANGE THIS to your filename
    
    if not os.path.exists(stl_filename):
        print(f"Please rename '{stl_filename}' in the script to your actual file.")
    else:
        # 1. Load Base Anatomy ONCE
        base_mesh, base_cl = load_and_skeletonize(stl_filename, voxel_pitch=0.5)
        print(f"Base anatomy loaded. {len(base_cl)} centerline points.")

        # 2. Generate 50 Separate Samples
        dataset = []
        num_samples = 50
        print(f"\nGenerating {num_samples} localized samples...")
        
        for i in range(num_samples):
            # Apply deformation to the BASE (fresh start every time)
            def_mesh, def_cl, meta = apply_single_localized_deformation(
                base_mesh, base_cl, 
                force_scale=12.0,  # Increased Magnitude (was 5.0)
                sigma=5.0          # Influence Radius (mm)
            )
            
            # Store everything
            sample = {
                "id": i,
                "mesh": def_mesh,       # The 3D Mesh (for X-ray projection)
                "centerline": def_cl,   # The Graph (for Physics Loss)
                "label": meta           # The Ground Truth
            }
            dataset.append(sample)
            print(f"  - Sample {i}: Deformed at {np.round(meta['epicenter_xyz'], 1)}")

        # 3. Visualize 3 Random Samples
        print("\nVisualizing...")
        fig = plt.figure(figsize=(18, 6))
        
        # Pick 3 random indices
        indices_to_show = np.random.choice(num_samples, 3, replace=False)
        
        for plot_idx, sample_idx in enumerate(indices_to_show):
            sample = dataset[sample_idx]
            ax = fig.add_subplot(1, 3, plot_idx+1, projection='3d')
            
            # --- CHANGE IS HERE ---
            # Original = Light Blue
            ax.scatter(base_cl[:,0], base_cl[:,1], base_cl[:,2], 
                      c='deepskyblue', s=1, alpha=0.2, label='Original')
            
            # Deformed = Red
            def_cl = sample['centerline']
            ax.scatter(def_cl[:,0], def_cl[:,1], def_cl[:,2], 
                      c='red', s=2, alpha=0.8, label='Deformed')
            
            # Poke Site = Black Star
            epicenter = sample['label']['epicenter_xyz']
            ax.scatter(epicenter[0], epicenter[1], epicenter[2], 
                      c='black', marker='*', s=150, label='Poke Site')
            
            ax.set_title(f"Sample {sample['id']}")
            if plot_idx == 0: ax.legend()

        plt.tight_layout()
        plt.show()

        # 4. Save Meshes to Disk
        output_dir = "output_meshes"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"\nSaving {num_samples} meshes to '{output_dir}/'...")
        
        for sample in dataset:
            file_name = f"deformed_{sample['id']}.stl"
            full_path = os.path.join(output_dir, file_name)
            
            # Export using Trimesh built-in function
            sample['mesh'].export(full_path)
            print(f"  - Saved: {file_name}")
            
        print("Done!")

