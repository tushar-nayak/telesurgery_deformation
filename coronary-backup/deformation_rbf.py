import numpy as np
import trimesh
import os
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from skimage.morphology import skeletonize
import imageio.v2 as imageio
import shutil

# --- Physics Kernels (From your logic) ---
def gaussian_kernel(r, sigma=5.0):
    return np.exp(-r**2 / (2 * sigma**2))

# --- 1. Graph/Path Logic ---
def get_path_segment(centerline, length_window=50):
    total_points = len(centerline)
    if total_points < length_window:
        return np.arange(total_points)
    start_idx = np.random.randint(0, total_points - length_window)
    return np.arange(start_idx, start_idx + length_window)

# --- 2. The Core Hybrid Deformation Logic (Applied sequentially) ---
def apply_hybrid_sequential(mesh, centerline, output_dir, branch_id, pass_id, num_frames=15):
    """
    Applies the Hybrid (RBF + Gaussian) logic in a sliding window fashion.
    """
    N = len(centerline)
    path_indices = get_path_segment(centerline, length_window=int(N/2))
    
    # Static Control Points for the whole vessel (Sparse grid)
    num_cp = 15
    cp_indices = np.linspace(0, N-1, num_cp, dtype=int)
    control_points_rest = centerline[cp_indices]
    
    # Directories
    viz_dir = os.path.join(output_dir, "viz_frames")
    if not os.path.exists(viz_dir): os.makedirs(viz_dir)
    
    gif_frames = []
    step_size = max(1, len(path_indices) // num_frames)
    
    # Base "Global" Force that evolves slowly
    global_force_bias = np.random.uniform(-3, 3, size=3)
    
    print(f"  > Simulating Hybrid Branch {branch_id}, Pass {pass_id}...")
    
    frame_count = 0
    for t in range(0, len(path_indices), step_size):
        # 1. Identify "Active Region" (Where the wire is)
        tip_idx_global = path_indices[t]
        tip_pos = centerline[tip_idx_global]
        
        # 2. GENERATE SPARSE DISPLACEMENTS (Hybrid Logic Step 1 & 2)
        # Instead of completely random -5 to 5 everywhere, we keep it 
        # localized to the tip to simulate sequential movement, 
        # but we use the "Sparse Control Point" idea.
        
        cp_displacements = np.zeros_like(control_points_rest)
        dists_to_tip = np.linalg.norm(control_points_rest - tip_pos, axis=1)
        
        # Influence radius for the "Main Bend"
        radius = 20.0 
        weights = np.exp(-dists_to_tip**2 / (2 * (radius/2)**2))
        
        # Apply randomized force to active control points
        # This matches "np.random.uniform(-5, 5)" but weighted by location
        local_randomness = np.random.uniform(-2, 2, size=cp_displacements.shape)
        
        for i in range(num_cp):
            if weights[i] > 0.01:
                # Base drift + Local random chaos
                force = (global_force_bias + local_randomness[i]) * weights[i]
                cp_displacements[i] = force
        
        # 3. RBF INTERPOLATION (Hybrid Logic Step 3)
        # Using 'thin_plate' as requested
        try:
            rbf_x = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], 
                        cp_displacements[:,0], function='thin_plate', smooth=0.1)
            rbf_y = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], 
                        cp_displacements[:,1], function='thin_plate', smooth=0.1)
            rbf_z = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], 
                        cp_displacements[:,2], function='thin_plate', smooth=0.1)
        except np.linalg.LinAlgError:
             print("    ! RBF SVD did not converge, skipping frame.")
             continue

        # Apply to Centerline
        dx_cl = rbf_x(centerline[:,0], centerline[:,1], centerline[:,2])
        dy_cl = rbf_y(centerline[:,0], centerline[:,1], centerline[:,2])
        dz_cl = rbf_z(centerline[:,0], centerline[:,1], centerline[:,2])
        
        dense_displacement = np.stack([dx_cl, dy_cl, dz_cl], axis=1)
        
        # Apply to Mesh (Visualization only if speed is needed, but we need it for export)
        verts = mesh.vertices
        dx_m = rbf_x(verts[:,0], verts[:,1], verts[:,2])
        dy_m = rbf_y(verts[:,0], verts[:,1], verts[:,2])
        dz_m = rbf_z(verts[:,0], verts[:,1], verts[:,2])
        mesh_disp = np.stack([dx_m, dy_m, dz_m], axis=1)

        # 4. LOCAL GAUSSIAN BUMP (Hybrid Logic Step 4)
        # "Fine detail or kinks that RBF might smooth out too much"
        # We add this bump explicitly near the TIP position
        if np.random.rand() > 0.3: # 70% chance of a kink
            kink_force = np.random.uniform(-2, 2, size=3)
            
            # Weighted by distance to tip (so kinks happen where the wire is)
            dists_cl = np.linalg.norm(centerline - tip_pos, axis=1)
            w_cl = gaussian_kernel(dists_cl, sigma=3.0) # Sharp kink (sigma=3)
            gaussian_disp_cl = np.outer(w_cl, kink_force)
            
            dists_mesh = np.linalg.norm(verts - tip_pos, axis=1)
            w_mesh = gaussian_kernel(dists_mesh, sigma=3.0)
            gaussian_disp_mesh = np.outer(w_mesh, kink_force)
            
            dense_displacement += gaussian_disp_cl
            mesh_disp += gaussian_disp_mesh

        # Final Deformed State
        deformed_cl = centerline + dense_displacement
        new_mesh = mesh.copy()
        new_mesh.vertices += mesh_disp

        # --- SAVE DATA ---
        file_prefix = f"branch_{branch_id:02d}_pass_{pass_id:02d}_frame_{frame_count:03d}"
        
        np.save(os.path.join(output_dir, f"{file_prefix}_centerline.npy"), deformed_cl)
        new_mesh.export(os.path.join(output_dir, f"{file_prefix}_mesh.stl"))
        
        # Important: Save the SPARSE parameters so the network can learn them
        np.save(os.path.join(output_dir, f"{file_prefix}_sparse_disp.npy"), cp_displacements)
        np.save(os.path.join(output_dir, f"{file_prefix}_cp_indices.npy"), cp_indices)

        # --- DEBUG VISUALIZATION ---
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Rest (Blue Faint)
        ax.plot(centerline[:,0], centerline[:,1], centerline[:,2], c='blue', alpha=0.1, linewidth=1, label="Rest")
        
        # Plot Deformed (Red Strong)
        ax.plot(deformed_cl[:,0], deformed_cl[:,1], deformed_cl[:,2], c='red', linewidth=2, label="Hybrid Deform")
        
        # Plot Control Points (Green Dots) & Forces (Arrows)
        # We plot the ACTIVE control points larger
        active_mask = np.linalg.norm(cp_displacements, axis=1) > 0.1
        
        # Inactive CPs
        ax.scatter(control_points_rest[~active_mask,0], control_points_rest[~active_mask,1], control_points_rest[~active_mask,2], c='green', s=10, alpha=0.3)
        
        # Active CPs + Forces
        if np.any(active_mask):
            active_cps = control_points_rest[active_mask]
            active_forces = cp_displacements[active_mask]
            ax.scatter(active_cps[:,0], active_cps[:,1], active_cps[:,2], c='green', s=50, edgecolors='black')
            ax.quiver(active_cps[:,0], active_cps[:,1], active_cps[:,2], 
                      active_forces[:,0], active_forces[:,1], active_forces[:,2], 
                      color='orange', length=1.0, label="RBF Forces")

        ax.set_title(f"Hybrid Sim | Br {branch_id} | Fr {frame_count}")
        ax.legend()
        
        frame_path = os.path.join(viz_dir, f"viz_{frame_count:03d}.png")
        plt.savefig(frame_path)
        plt.close()
        gif_frames.append(imageio.imread(frame_path))
        
        frame_count += 1
        if frame_count >= num_frames: break

    # Create GIF
    gif_path = os.path.join(output_dir, f"debug_branch_{branch_id:02d}_pass_{pass_id:02d}.gif")
    imageio.mimsave(gif_path, gif_frames, fps=5)
    print(f"    -> Visualized: {gif_path}")
    shutil.rmtree(viz_dir)

# --- Main Execution ---
if __name__ == "__main__":
    input_stl = "vessel.stl" 
    output_dir = "output_meshes_sequential_hybrid"
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    print("Loading mesh...")
    mesh = trimesh.load(input_stl)
    
    # Centerline Extraction
    voxel_grid = mesh.voxelized(pitch=0.5).fill()
    skel = skeletonize(voxel_grid.matrix)
    centerline = voxel_grid.indices_to_points(np.argwhere(skel))
    centerline = centerline[np.argsort(centerline[:, 2])]
    
    # Run
    NUM_BRANCHES = 3      
    PASSES_PER_BRANCH = 2 
    FRAMES_PER_PASS = 10  
    
    for b in range(NUM_BRANCHES):
        for p in range(PASSES_PER_BRANCH):
            apply_hybrid_sequential(mesh, centerline, output_dir, b, p, FRAMES_PER_PASS)
            
    print("\nDone! Check 'output_meshes_sequential_hybrid' for data and GIFs.")