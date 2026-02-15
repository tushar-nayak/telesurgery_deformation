import numpy as np
import trimesh
import os
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from skimage.morphology import skeletonize
import imageio.v2 as imageio  # For creating GIFs
import shutil

# --- Physics Kernels ---
def thin_plate_spline(r):
    if r == 0: return 0
    return (r**2) * np.log(r + 1e-6)

# --- 1. Graph/Path Logic ---
def get_path_segment(centerline, length_window=50):
    total_points = len(centerline)
    if total_points < length_window:
        return np.arange(total_points)
    start_idx = np.random.randint(0, total_points - length_window)
    return np.arange(start_idx, start_idx + length_window)

# --- 2. Sequential Deformation with Visualization ---
def apply_sequential_deformation(mesh, centerline, output_dir, branch_id, pass_id, num_frames=20):
    """
    Simulates deformation and creates a debug GIF for this specific pass.
    """
    N = len(centerline)
    path_indices = get_path_segment(centerline, length_window=int(N/2))
    
    # Static Control Points (The "Sensors")
    num_cp = 15
    cp_indices = np.linspace(0, N-1, num_cp, dtype=int)
    control_points_rest = centerline[cp_indices]
    
    # Base Force
    base_force = np.random.uniform(-4, 4, size=3)
    
    # Temp folder for GIF frames
    viz_dir = os.path.join(output_dir, "viz_frames")
    if not os.path.exists(viz_dir): os.makedirs(viz_dir)
    
    gif_frames = []
    step_size = max(1, len(path_indices) // num_frames)
    
    print(f"  > Simulating Branch {branch_id}, Pass {pass_id}...")

    frame_count = 0
    for t in range(0, len(path_indices), step_size):
        tip_idx_global = path_indices[t]
        tip_pos = centerline[tip_idx_global]
        
        # Physics: Apply Force
        noise = np.random.normal(0, 0.5, size=3)
        current_force = base_force + noise
        
        cp_displacements = np.zeros_like(control_points_rest)
        dists_to_tip = np.linalg.norm(control_points_rest - tip_pos, axis=1)
        radius = 15.0 
        weights = np.exp(-dists_to_tip**2 / (2 * (radius/2)**2))
        
        for i in range(num_cp):
            cp_displacements[i] = current_force * weights[i]
            
        # Physics: RBF Interpolation
        rbf_x = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], 
                    cp_displacements[:,0], function='thin_plate', smooth=0.5)
        rbf_y = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], 
                    cp_displacements[:,1], function='thin_plate', smooth=0.5)
        rbf_z = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], 
                    cp_displacements[:,2], function='thin_plate', smooth=0.5)
        
        dx_cl = rbf_x(centerline[:,0], centerline[:,1], centerline[:,2])
        dy_cl = rbf_y(centerline[:,0], centerline[:,1], centerline[:,2])
        dz_cl = rbf_z(centerline[:,0], centerline[:,1], centerline[:,2])
        
        deformed_cl = centerline + np.stack([dx_cl, dy_cl, dz_cl], axis=1)
        
        # Deform Mesh (Optional for visualization speedup, but needed for data)
        # Note: Doing full mesh deformation every frame is slow. For training data we need it.
        verts = mesh.vertices
        dx_m = rbf_x(verts[:,0], verts[:,1], verts[:,2])
        dy_m = rbf_y(verts[:,0], verts[:,1], verts[:,2])
        dz_m = rbf_z(verts[:,0], verts[:,1], verts[:,2])
        new_mesh = mesh.copy()
        new_mesh.vertices += np.stack([dx_m, dy_m, dz_m], axis=1)
        
        # --- SAVE DATA (Improved Naming) ---
        # Format: branch_XX_pass_XX_frame_XXX
        file_prefix = f"branch_{branch_id:02d}_pass_{pass_id:02d}_frame_{frame_count:03d}"
        
        np.save(os.path.join(output_dir, f"{file_prefix}_centerline.npy"), deformed_cl)
        new_mesh.export(os.path.join(output_dir, f"{file_prefix}_mesh.stl"))
        np.save(os.path.join(output_dir, f"{file_prefix}_sparse_disp.npy"), cp_displacements)
        np.save(os.path.join(output_dir, f"{file_prefix}_cp_indices.npy"), cp_indices)

        # --- DEBUG VISUALIZATION (Create Plot) ---
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Plot Rest State (Blue Faint)
        ax.plot(centerline[:,0], centerline[:,1], centerline[:,2], c='blue', alpha=0.1, linewidth=1, label="Rest State")
        
        # 2. Plot Deformed State (Red Strong)
        ax.plot(deformed_cl[:,0], deformed_cl[:,1], deformed_cl[:,2], c='red', linewidth=2, label="Deformed")
        
        # 3. Plot The Force Location (Yellow Star)
        ax.scatter([tip_pos[0]], [tip_pos[1]], [tip_pos[2]], c='orange', s=100, marker='*', label="Guidewire Tip")
        
        # 4. Plot Control Point Forces (Green Arrows)
        # Only plot vectors that have significant magnitude
        for i in range(num_cp):
            if np.linalg.norm(cp_displacements[i]) > 0.1:
                start = control_points_rest[i]
                vec = cp_displacements[i]
                ax.quiver(start[0], start[1], start[2], vec[0], vec[1], vec[2], color='green', length=1.0)

        ax.set_title(f"Branch {branch_id} | Pass {pass_id} | Frame {frame_count}")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend()
        
        # Save Frame
        frame_path = os.path.join(viz_dir, f"viz_{frame_count:03d}.png")
        plt.savefig(frame_path)
        plt.close()
        gif_frames.append(imageio.imread(frame_path))
        
        frame_count += 1
        if frame_count >= num_frames: break

    # Create GIF for this pass
    gif_path = os.path.join(output_dir, f"debug_branch_{branch_id:02d}_pass_{pass_id:02d}.gif")
    imageio.mimsave(gif_path, gif_frames, fps=5)
    print(f"    -> Created Visualization: {gif_path}")
    
    # Cleanup temp frames
    shutil.rmtree(viz_dir)

# --- Main Execution ---
if __name__ == "__main__":
    input_stl = "vessel.stl" 
    output_dir = "output_meshes_sequential"
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    print("Loading mesh...")
    mesh = trimesh.load(input_stl)
    
    # Centerline Logic
    voxel_grid = mesh.voxelized(pitch=0.5).fill()
    skel = skeletonize(voxel_grid.matrix)
    centerline = voxel_grid.indices_to_points(np.argwhere(skel))
    centerline = centerline[np.argsort(centerline[:, 2])]
    
    # SETTINGS
    NUM_BRANCHES = 3      # Reduced for demo speed (increase later)
    PASSES_PER_BRANCH = 2 
    FRAMES_PER_PASS = 10  
    
    print(f"Generating {NUM_BRANCHES * PASSES_PER_BRANCH} Sequences...")
    
    for b in range(NUM_BRANCHES):
        for p in range(PASSES_PER_BRANCH):
            apply_sequential_deformation(mesh, centerline, output_dir, branch_id=b, pass_id=p, num_frames=FRAMES_PER_PASS)

    print("\nDone! Check the .gif files in the output folder to see the deformations.")