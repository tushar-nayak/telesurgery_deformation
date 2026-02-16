import numpy as np
import trimesh
import os
import json
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from skimage.morphology import skeletonize
import imageio.v2 as imageio
import shutil
import time

# --- 1. CONFIGURATION (Tweak these for "Neuro" feel) ---
CONFIG = {
    # Data Gen Settings
    "NUM_BRANCHES": 1,          # How many random subsections to simulate
    "PASSES_PER_BRANCH": 3,     # How many times to simulate each section
    "FRAMES_PER_PASS": 40,      # Duration of simulation
    
    # Physics Settings
    "FORCE_MAGNITUDE": 2.5,     # Strength of the guidewire push (mm)
    "FORCE_DRIFT": 0.3,         # How fast the force direction changes (Lower = Stiffer wire)
    "INFLUENCE_RADIUS": 15.0,   # How much of the vessel moves around the tip
    "RBF_SMOOTHING": 0.8,       # 0.0 = Flexible/Jelly, 1.0+ = Stiff/Plate
    
    # Biological Realism
    "RELAXATION_FACTOR": 0.90,  # 1.0 = Elastic (Snaps back), 0.0 = Plastic (Stays deformed)
                                # 0.90 means it retains 90% of deformation per frame.
}

# --- 2. PHYSICS KERNELS ---
def thin_plate_spline(r):
    """Energy minimizing spline kernel (stiff material)"""
    if r == 0: return 0
    return (r**2) * np.log(r + 1e-6)

# --- 3. CORE LOGIC ---

def extract_centerline(mesh):
    """
    Your original skeletonization logic.
    """
    print("  > Voxelizing and Skeletonizing mesh...")
    # 1. Voxelize
    # Note: pitch=0.5 is high res. If slow, increase to 1.0
    voxel_grid = mesh.voxelized(pitch=0.5).fill()
    
    # 2. Skeletonize
    skel = skeletonize(voxel_grid.matrix)
    
    # 3. Convert to Points
    centerline = voxel_grid.indices_to_points(np.argwhere(skel))
    
    # 4. Sort (Basic Z-sort per your logic)
    # Note: For complex looping vessels, Graph-based sorting is better, 
    # but we stick to your logic here.
    centerline = centerline[np.argsort(centerline[:, 2])]
    
    print(f"  > Centerline extracted: {len(centerline)} points")
    return centerline

def get_path_segment(centerline, length_window=60):
    """Selects a random subsection of the vessel to simulate."""
    total_points = len(centerline)
    if total_points < length_window:
        return np.arange(total_points)
    
    # Pick a random start point
    start_idx = np.random.randint(0, total_points - length_window)
    return np.arange(start_idx, start_idx + length_window)

def apply_cumulative_deformation(mesh, centerline, output_dir, branch_id, pass_id, report_data):
    """
    Runs the Viscoelastic Cumulative Simulation.
    """
    # 1. Setup Simulation Segment
    N = len(centerline)
    path_indices = get_path_segment(centerline, length_window=int(N/2)) # Use half the vessel length
    
    # Control Points (The "Spine" handles) - Subset of centerline
    num_cp = 20
    cp_indices = np.linspace(0, N-1, num_cp, dtype=int)
    control_points_rest = centerline[cp_indices]
    
    # 2. Initialize State (Mutable)
    current_mesh_verts = mesh.vertices.copy()
    current_centerline = centerline.copy()
    
    # Force Vector Initialization (Random Direction)
    current_force_vec = np.random.normal(0, 1, size=3)
    current_force_vec /= np.linalg.norm(current_force_vec)
    current_force_vec *= CONFIG["FORCE_MAGNITUDE"]
    
    # Directories
    viz_dir = os.path.join(output_dir, "viz_frames")
    if os.path.exists(viz_dir): shutil.rmtree(viz_dir)
    os.makedirs(viz_dir)
    
    gif_frames = []
    pass_stats = {"id": f"b{branch_id}_p{pass_id}", "frames": []}
    original_edges = mesh.edges_unique_length # For strain calc

    print(f"  > Simulating Branch {branch_id}, Pass {pass_id}...")

    # 3. Simulation Loop
    step_size = max(1, len(path_indices) // CONFIG["FRAMES_PER_PASS"])
    
    frame_count = 0
    for t in range(0, len(path_indices), step_size):
        # A. Update Tip Position
        # CRITICAL: Tip location comes from the DEFORMED centerline
        tip_idx_global = path_indices[t]
        tip_pos = current_centerline[tip_idx_global]
        
        # B. Evolve Force (Random Walk)
        # Add drift to direction (smooth change)
        drift = np.random.normal(0, CONFIG["FORCE_DRIFT"], size=3)
        current_force_vec += drift
        
        # Clamp magnitude (Prevent physics explosion)
        mag = np.linalg.norm(current_force_vec)
        target_mag = CONFIG["FORCE_MAGNITUDE"]
        # Soft elasticity for the force magnitude itself
        current_force_vec = current_force_vec * (target_mag / mag) 
        
        # C. Calculate Displacement Influence
        # Distance from *rest* control points to *current* tip 
        # (Using rest CP for RBF stability is a standard trick)
        dists_to_tip = np.linalg.norm(control_points_rest - tip_pos, axis=1)
        weights = np.exp(-dists_to_tip**2 / (2 * (CONFIG["INFLUENCE_RADIUS"]/2)**2))
        
        # Create sparse displacement deltas for control points
        cp_deltas = np.zeros_like(control_points_rest)
        for i in range(num_cp):
            cp_deltas[i] = current_force_vec * weights[i]
            
        # D. RBF Interpolation (Sparse -> Dense)
        rbf_args = dict(function='thin_plate', smooth=CONFIG["RBF_SMOOTHING"])
        try:
            rbf_x = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], cp_deltas[:,0], **rbf_args)
            rbf_y = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], cp_deltas[:,1], **rbf_args)
            rbf_z = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], cp_deltas[:,2], **rbf_args)
        except np.linalg.LinAlgError:
            print("    [Warning] Singular Matrix in RBF. Skipping frame.")
            continue

        # E. Apply Incremental Deformation
        # Calculate moves for centerline and mesh
        d_cl_x = rbf_x(current_centerline[:,0], current_centerline[:,1], current_centerline[:,2])
        d_cl_y = rbf_y(current_centerline[:,0], current_centerline[:,1], current_centerline[:,2])
        d_cl_z = rbf_z(current_centerline[:,0], current_centerline[:,1], current_centerline[:,2])
        
        d_m_x = rbf_x(current_mesh_verts[:,0], current_mesh_verts[:,1], current_mesh_verts[:,2])
        d_m_y = rbf_y(current_mesh_verts[:,0], current_mesh_verts[:,1], current_mesh_verts[:,2])
        d_m_z = rbf_z(current_mesh_verts[:,0], current_mesh_verts[:,1], current_mesh_verts[:,2])
        
        # Update Positions
        current_centerline += np.stack([d_cl_x, d_cl_y, d_cl_z], axis=1)
        current_mesh_verts += np.stack([d_m_x, d_m_y, d_m_z], axis=1)
        
        # F. Viscoelastic Relaxation
        # Calculate vectors pointing back to rest state
        relax_vec_cl = centerline - current_centerline
        relax_vec_m = mesh.vertices - current_mesh_verts
        
        # Apply relaxation (Pull back by (1 - Factor))
        current_centerline += relax_vec_cl * (1.0 - CONFIG["RELAXATION_FACTOR"])
        current_mesh_verts += relax_vec_m * (1.0 - CONFIG["RELAXATION_FACTOR"])

        # --- SAVE & METRICS ---
        
        # Temporary mesh for saving/strain calc
        temp_mesh = mesh.copy()
        temp_mesh.vertices = current_mesh_verts
        
        # Calculate Strain (Approximation)
        # We only check max stretch to detect spaghettification
        new_edges = temp_mesh.edges_unique_length
        # safe divide
        ratios = new_edges / (original_edges + 1e-6)
        max_strain = np.max(ratios)
        mean_disp = np.mean(np.linalg.norm(current_mesh_verts - mesh.vertices, axis=1))

        # Log Stats
        pass_stats["frames"].append({
            "frame": frame_count,
            "max_strain": float(max_strain),
            "mean_disp": float(mean_disp),
            "force_vec": current_force_vec.tolist()
        })
        
        # Save Files
        file_prefix = f"branch_{branch_id:02d}_pass_{pass_id:02d}_frame_{frame_count:03d}"
        np.save(os.path.join(output_dir, f"{file_prefix}_centerline.npy"), current_centerline)
        temp_mesh.export(os.path.join(output_dir, f"{file_prefix}_mesh.stl"))
        
        # --- VISUALIZATION ---
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Rest State (Ghost)
        ax.plot(centerline[:,0], centerline[:,1], centerline[:,2], c='gray', alpha=0.3, linewidth=1, label="Rest")
        
        # 2. Current State (Red)
        ax.plot(current_centerline[:,0], current_centerline[:,1], current_centerline[:,2], c='red', linewidth=2, label="Deformed")
        
        # 3. Tip & Force
        ax.scatter([tip_pos[0]], [tip_pos[1]], [tip_pos[2]], c='orange', s=100, marker='*')
        # Draw Force Vector
        ax.quiver(tip_pos[0], tip_pos[1], tip_pos[2], 
                  current_force_vec[0], current_force_vec[1], current_force_vec[2], 
                  color='green', length=2.0, label="Force")

        # Stats Overlay
        ax.text2D(0.05, 0.95, f"Strain: {max_strain:.2f}x\nRel. Factor: {CONFIG['RELAXATION_FACTOR']}", 
                  transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # Keep camera steady (centered on middle of vessel)
        mid = np.mean(centerline, axis=0)
        rng = 30 # Zoom level
        ax.set_xlim(mid[0]-rng, mid[0]+rng); ax.set_ylim(mid[1]-rng, mid[1]+rng); ax.set_zlim(mid[2]-rng, mid[2]+rng)
        
        ax.set_title(f"Viscoelastic Sim | Frame {frame_count}")
        
        frame_path = os.path.join(viz_dir, f"viz_{frame_count:03d}.png")
        plt.savefig(frame_path)
        plt.close()
        gif_frames.append(imageio.imread(frame_path))
        
        frame_count += 1
        if frame_count >= CONFIG["FRAMES_PER_PASS"]: break

    # Create GIF
    gif_path = os.path.join(output_dir, f"neuro_sim_branch_{branch_id}_pass_{pass_id}.gif")
    imageio.mimsave(gif_path, gif_frames, fps=10)
    shutil.rmtree(viz_dir)
    
    report_data["simulations"].append(pass_stats)

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    input_stl = "vessel.stl" 
    output_dir = "output_temporal_one"
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # --- LOAD MESH ---
    if not os.path.exists(input_stl):
        print("[INFO] 'vessel.stl' not found. Generating a dummy tube for demonstration...")
        # Create a helical/spiral path to mimic a neuro vessel
        t = np.linspace(0, 4*np.pi, 100)
        x = 10 * np.cos(t)
        y = 10 * np.sin(t)
        z = np.linspace(0, 60, 100)
        path = np.column_stack((x, y, z))
        mesh = trimesh.creation.cylinder(radius=2.5, segment=path, sections=16)
        mesh.export(input_stl)
    else:
        print(f"[INFO] Loading {input_stl}...")
        mesh = trimesh.load(input_stl)

    # --- SKELETONIZE (Your Logic) ---
    centerline = extract_centerline(mesh)
    
    # --- RUN SIMULATION ---
    report_data = {
        "config": CONFIG,
        "mesh_info": {"verts": len(mesh.vertices), "faces": len(mesh.faces)},
        "simulations": []
    }
    
    print(f"Starting Simulation of {CONFIG['NUM_BRANCHES'] * CONFIG['PASSES_PER_BRANCH']} Sequences...")
    
    for b in range(CONFIG["NUM_BRANCHES"]):
        for p in range(CONFIG["PASSES_PER_BRANCH"]):
            apply_cumulative_deformation(mesh, centerline, output_dir, b, p, report_data)

    # --- SAVE REPORT ---
    report_path = os.path.join(output_dir, "simulation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
        
    print("\n" + "="*50)
    print("SIMULATION COMPLETE")
    print(f"1. Check GIFs in: {output_dir}")
    print(f"2. Upload this file for analysis: {report_path}")
    print("="*50)
