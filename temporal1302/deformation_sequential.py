import numpy as np
import trimesh
import os
from scipy.interpolate import Rbf
from skimage.morphology import skeletonize

# --- Physics Kernels ---
def thin_plate_spline(r):
    if r == 0: return 0
    return (r**2) * np.log(r + 1e-6)

# --- 1. Graph/Path Logic ---
def get_path_segment(centerline, length_window=50):
    """
    Selects a continuous segment of the vessel to simulate a single branch traversal.
    """
    total_points = len(centerline)
    if total_points < length_window:
        return np.arange(total_points) # Return whole thing if small
    
    # Pick a random start point
    start_idx = np.random.randint(0, total_points - length_window)
    end_idx = start_idx + length_window
    
    # Return indices
    return np.arange(start_idx, end_idx)

# --- 2. Sequential Deformation Engine ---
def apply_sequential_deformation(mesh, centerline, output_dir, sequence_id, num_frames=20):
    """
    Simulates a guidewire moving through ONE branch over 'num_frames' steps.
    """
    N = len(centerline)
    
    # 1. Pick a "Branch" (Path Segment)
    # We will simulate the wire moving along this specific path
    path_indices = get_path_segment(centerline, length_window=int(N/2)) # Use half the vessel length
    path_points = centerline[path_indices]
    
    # 2. Define Control Points for the WHOLE mesh (Static grid)
    # We use these to drive the RBF
    num_cp = 15
    cp_indices = np.linspace(0, N-1, num_cp, dtype=int)
    control_points_rest = centerline[cp_indices]
    
    # 3. Pre-Generate a "Force Profile"
    # The force shouldn't change randomly every millimeter. It should change smoothly.
    # We create a random force vector that slowly rotates/changes as the wire moves.
    base_force = np.random.uniform(-4, 4, size=3)
    
    # 4. Run the Sequence (Time Steps)
    # We interpolate positions along the path
    step_size = len(path_indices) // num_frames
    if step_size < 1: step_size = 1
    
    frame_count = 0
    for t in range(0, len(path_indices), step_size):
        # Current position of the "Guidewire Tip"
        tip_idx_global = path_indices[t]
        tip_pos = centerline[tip_idx_global]
        
        # --- A. Calculate Force for this Frame ---
        # Slowly evolve the force vector (Simulate wire twisting)
        noise = np.random.normal(0, 0.5, size=3)
        current_force = base_force + noise
        
        # --- B. Perturb Control Points (Localized) ---
        # Only control points NEAR the tip should move. 
        # Others stay at 0 displacement (Return to Normal)
        cp_displacements = np.zeros_like(control_points_rest)
        
        dists_to_tip = np.linalg.norm(control_points_rest - tip_pos, axis=1)
        
        # Influence radius: e.g., 15mm around the tip
        radius = 15.0 
        
        # Gaussian Falloff for smoothness
        weights = np.exp(-dists_to_tip**2 / (2 * (radius/2)**2))
        
        # Apply force weighted by distance
        for i in range(num_cp):
            cp_displacements[i] = current_force * weights[i]
            
        # --- C. Solve RBF (Global Deformation) ---
        # Fit RBF: f(rest_pos) -> displacement
        # Using TPS ensures smoothness
        rbf_x = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], 
                    cp_displacements[:,0], function='thin_plate', smooth=0.5)
        rbf_y = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], 
                    cp_displacements[:,1], function='thin_plate', smooth=0.5)
        rbf_z = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], 
                    cp_displacements[:,2], function='thin_plate', smooth=0.5)
        
        # Deform Centerline
        dx_cl = rbf_x(centerline[:,0], centerline[:,1], centerline[:,2])
        dy_cl = rbf_y(centerline[:,0], centerline[:,1], centerline[:,2])
        dz_cl = rbf_z(centerline[:,0], centerline[:,1], centerline[:,2])
        
        deformed_cl = centerline + np.stack([dx_cl, dy_cl, dz_cl], axis=1)
        
        # Deform Mesh
        verts = mesh.vertices
        dx_m = rbf_x(verts[:,0], verts[:,1], verts[:,2])
        dy_m = rbf_y(verts[:,0], verts[:,1], verts[:,2])
        dz_m = rbf_z(verts[:,0], verts[:,1], verts[:,2])
        
        new_mesh = mesh.copy()
        new_mesh.vertices += np.stack([dx_m, dy_m, dz_m], axis=1)
        
        # --- D. Save Data ---
        # Naming: seq_{ID}_frame_{T}
        save_base = f"seq_{sequence_id}_frame_{frame_count:03d}"
        
        # Save Geometry
        np.save(os.path.join(output_dir, f"{save_base}_centerline.npy"), deformed_cl)
        new_mesh.export(os.path.join(output_dir, f"{save_base}_mesh.stl"))
        
        # Save Parameters (Targets for training)
        # We save the SPARSE displacements (The network tries to predict these)
        np.save(os.path.join(output_dir, f"{save_base}_sparse_disp.npy"), cp_displacements)
        np.save(os.path.join(output_dir, f"{save_base}_cp_indices.npy"), cp_indices)
        
        print(f"    Saved Frame {frame_count} (Tip at index {tip_idx_global})")
        frame_count += 1
        
        if frame_count >= num_frames: break

# --- Main Execution ---
if __name__ == "__main__":
    input_stl = "vessel.stl" 
    output_dir = "output_meshes_sequential"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Loading mesh...")
    mesh = trimesh.load(input_stl)
    
    # Fast centerline (Replace with your robust loader if available)
    voxel_grid = mesh.voxelized(pitch=0.5).fill()
    skel = skeletonize(voxel_grid.matrix)
    centerline = voxel_grid.indices_to_points(np.argwhere(skel))
    # Simple sort to make it line-like
    centerline = centerline[np.argsort(centerline[:, 2])]
    
    # SETTINGS
    NUM_BRANCHES = 5   # How many different "paths" to simulate
    PASSES_PER_BRANCH = 3 # How many times to run the wire through each path
    FRAMES_PER_PASS = 15  # How many steps in the animation
    
    print(f"Starting Simulation:")
    print(f"  - {NUM_BRANCHES} Branches")
    print(f"  - {PASSES_PER_BRANCH} Passes each")
    print(f"  - {FRAMES_PER_PASS} Frames per pass")
    print(f"  = {NUM_BRANCHES * PASSES_PER_BRANCH * FRAMES_PER_PASS} Total Training Samples\n")
    
    seq_counter = 0
    
    for b in range(NUM_BRANCHES):
        print(f"Simulating Branch {b}...")
        
        for p in range(PASSES_PER_BRANCH):
            print(f"  Pass {p} (Sequence {seq_counter})...")
            
            # This function runs the sliding window logic and saves files
            apply_sequential_deformation(mesh, centerline, output_dir, seq_counter, num_frames=FRAMES_PER_PASS)
            
            seq_counter += 1
            
    print("\nDone! Data generated in 'output_meshes_sequential'.")