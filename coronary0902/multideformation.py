import numpy as np
import trimesh
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import copy

# --- 1. Load STL & Extract Centerline (Same as before) ---
def load_and_skeletonize(stl_path, voxel_pitch=0.5):
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"Could not find file: {stl_path}")

    print(f"Loading {stl_path}...")
    mesh = trimesh.load_mesh(stl_path)
    print("Voxelizing...")
    voxel_grid = mesh.voxelized(pitch=voxel_pitch).fill()
    print("Skeletonizing...")
    skeleton = skeletonize(voxel_grid.matrix)
    indices = np.argwhere(skeleton)
    centerline_points = voxel_grid.indices_to_points(indices)

    # Sort points
    if len(centerline_points) > 0:
        sorted_indices = [0]
        remaining = list(range(1, len(centerline_points)))
        while remaining:
            last_pt = centerline_points[sorted_indices[-1]]
            dists = np.linalg.norm(centerline_points[remaining] - last_pt, axis=1)
            nearest = np.argmin(dists)
            sorted_indices.append(remaining[nearest])
            remaining.pop(nearest)
        centerline_points = centerline_points[sorted_indices]
        
    return mesh, centerline_points

# --- 2. LOCALIZED Gaussian Deformation Logic ---
def gaussian_kernel(r, sigma=5.0):
    """
    Gaussian Kernel: decays to zero quickly. 
    Controls "locality" - only points within ~3*sigma will move.
    """
    return np.exp(-r**2 / (2 * sigma**2))

def apply_single_localized_deformation(original_mesh, original_centerline, 
                                       force_scale=8.0, # Increased magnitude
                                       sigma=4.0):      # Radius of influence
    """
    Creates ONE copy of the vessel with ONE localized deformation.
    """
    # Create deep copies so we don't mess up the original for the next loop
    mesh = original_mesh.copy()
    centerline = original_centerline.copy()
    
    # 1. Pick ONE random point on the centerline to be the "Epicenter"
    center_idx = np.random.randint(0, len(centerline))
    epicenter = centerline[center_idx]
    
    # 2. Generate ONE strong random force vector
    force = np.random.uniform(-1, 1, size=3)
    force = (force / np.linalg.norm(force)) * force_scale
    
    # --- A. Deform Mesh (Gaussian Weighting) ---
    # Calc distance from every vertex to the epicenter
    dists_mesh = np.linalg.norm(mesh.vertices - epicenter, axis=1)
    
    # Calculate weights: 1.0 at epicenter, 0.0 far away
    weights_mesh = gaussian_kernel(dists_mesh, sigma=sigma)
    
    # Apply displacement: weight * force
    # We reshape force to broadcast it across all vertices
    disp_mesh = np.outer(weights_mesh, force)
    mesh.vertices += disp_mesh
    
    # --- B. Deform Centerline ---
    dists_cl = np.linalg.norm(centerline - epicenter, axis=1)
    weights_cl = gaussian_kernel(dists_cl, sigma=sigma)
    disp_cl = np.outer(weights_cl, force)
    centerline += disp_cl
    
    # Return the deformed objects AND the metadata (where it happened)
    metadata = {
        "epicenter_index": center_idx,
        "epicenter_xyz": epicenter,
        "applied_force": force
    }
    
    return mesh, centerline, metadata

# --- Main Execution ---
if __name__ == "__main__":
    stl_filename = "vessel.stl" 
    
    if not os.path.exists(stl_filename):
        print(f"Please rename '{stl_filename}' to your actual file.")
    else:
        # 1. Load Original ONCE
        base_mesh, base_cl = load_and_skeletonize(stl_filename, voxel_pitch=0.5)
        print(f"Base loaded. Centerline points: {len(base_cl)}")

        # 2. Generate 20 Individual Samples
        dataset = []
        num_samples = 20
        
        print(f"\nGenerating {num_samples} localized deformation samples...")
        
        for i in range(num_samples):
            # Apply deformation to the BASE mesh (fresh start every time)
            def_mesh, def_cl, meta = apply_single_localized_deformation(
                base_mesh, base_cl, 
                force_scale=10.0,  # Increased Magnitude
                sigma=3.0         # Tweak this: Lower = sharper kink, Higher = wider bend
            )
            
            # Record the sample
            sample = {
                "id": i,
                "mesh": def_mesh,
                "centerline": def_cl,
                "label": meta # This tells you WHERE it was deformed
            }
            dataset.append(sample)
            print(f"  - Sample {i}: Deformed at index {meta['epicenter_index']}")

        # 3. Visualize a few random ones to verify
        print("\nVisualizing 3 random samples...")
        fig = plt.figure(figsize=(15, 5))
        
        # Pick 3 random indices to show
        show_indices = np.random.choice(num_samples, 3, replace=False)
        
        for idx, sample_idx in enumerate(show_indices):
            sample = dataset[sample_idx]
            ax = fig.add_subplot(1, 3, idx+1, projection='3d')
            
            # Plot Original (Blue, small)
            ax.scatter(base_cl[:,0], base_cl[:,1], base_cl[:,2], c='blue', s=1, alpha=0.2, label='Original')
            
            # Plot Deformed (Red, thicker)
            # We highlight the specific area that moved
            deformed_cl = sample['centerline']
            ax.plot(deformed_cl[:,0], deformed_cl[:,1], deformed_cl[:,2], c='red', linewidth=2, label='Deformed')
            
            # Plot the "Poke" location (Black Star)
            epicenter = sample['label']['epicenter_xyz']
            ax.scatter(epicenter[0], epicenter[1], epicenter[2], c='black', marker='*', s=100, label='Poke Site')
            
            ax.set_title(f"Sample {sample['id']} (Idx: {sample['label']['epicenter_index']})")
            if idx == 0: ax.legend()

        plt.tight_layout()
        plt.show()

        # NOTE: You can now access dataset[0]['mesh'], dataset[0]['label'], etc.