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

# --- 1. CONFIGURATION ---
CONFIG = {
    # Data Gen Settings
    "NUM_BRANCHES": 5,          # How many distinct anatomical regions to test
    "PASSES_PER_BRANCH": 3,     # How many times to run the wire through THAT SAME branch
    "FRAMES_PER_PASS": 40,      # Duration of simulation
    
    # Physics Settings
    "FORCE_MAGNITUDE": 2.0,     # Strength of the guidewire push (mm)
    "FORCE_DRIFT": 0.3,         # How fast the force direction changes
    "INFLUENCE_RADIUS": 15.0,   # How much of the vessel moves around the tip
    "RBF_SMOOTHING": 0.8,       # 0.8 = Stiff wire, 0.1 = Jelly
    
    # Hybrid Settings
    "Gaussian_Kink_Prob": 0.3,  # 30% chance of a sharp kink per frame
    
    # Biological Realism
    "RELAXATION_FACTOR": 0.90   # 1.0 = Elastic (Snaps back), 0.9 = Viscoelastic
}

# --- 2. PHYSICS KERNELS ---
def thin_plate_spline(r):
    """Energy minimizing spline kernel (stiff material)"""
    if r == 0: return 0
    return (r**2) * np.log(r + 1e-6)

def gaussian_kernel(r, sigma=4.0):
    """Localized bump for sharp kinks"""
    return np.exp(-r**2 / (2 * sigma**2))

# --- 3. CORE LOGIC ---

def extract_centerline(mesh):
    """
    Robust skeletonization logic.
    """
    print("  > Voxelizing and Skeletonizing mesh...")
    # High-res voxelization (pitch=0.5). Increase pitch if too slow.
    voxel_grid = mesh.voxelized(pitch=0.5).fill()
    skel = skeletonize(voxel_grid.matrix)
    centerline = voxel_grid.indices_to_points(np.argwhere(skel))
    
    # Basic Z-sort. For production, use graph traversal if vessels loop back.
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

def apply_cumulative_hybrid_deformation(mesh, centerline, output_dir, branch_id, pass_id, report_data, path_indices):
    """
    Runs the Viscoelastic + Hybrid Simulation on a LOCKED path segment.
    """
    N = len(centerline)
    
    # 1. Use the LOCKED path indices passed from main()
    # This ensures every pass uses the same anatomy!
    
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
        # --- A. Update Tip Position ---
        # Tip location comes from the DEFORMED centerline (tracking the vessel wall)
        tip_idx_global = path_indices[t]
        tip_pos = current_centerline[tip_idx_global]
        
        # --- B. Evolve Force (Random Walk) ---
        drift = np.random.normal(0, CONFIG["FORCE_DRIFT"], size=3)
        current_force_vec += drift
        
        # Soft clamp magnitude
        mag = np.linalg.norm(current_force_vec)
        current_force_vec = current_force_vec * (CONFIG["FORCE_MAGNITUDE"] / mag) 
        
        # --- C. Hybrid Force Calculation ---
        
        # 1. RBF Forces (Global Bending)
        dists_to_tip = np.linalg.norm(control_points_rest - tip_pos, axis=1)
        weights = np.exp(-dists_to_tip**2 / (2 * (CONFIG["INFLUENCE_RADIUS"]/2)**2))
        
        cp_deltas = np.zeros_like(control_points_rest)
        for i in range(num_cp):
            cp_deltas[i] = current_force_vec * weights[i]
            
        # 2. RBF Interpolation (Sparse -> Dense)
        rbf_args = dict(function='thin_plate', smooth=CONFIG["RBF_SMOOTHING"])
        try:
            rbf_x = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], cp_deltas[:,0], **rbf_args)
            rbf_y = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], cp_deltas[:,1], **rbf_args)
            rbf_z = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], cp_deltas[:,2], **rbf_args)
        except np.linalg.LinAlgError:
            print("    [Warning] Singular Matrix in RBF. Skipping frame.")
            continue

        # Calculate Base Deformation
        d_cl_x = rbf_x(current_centerline[:,0], current_centerline[:,1], current_centerline[:,2])
        d_cl_y = rbf_y(current_centerline[:,0], current_centerline[:,1], current_centerline[:,2])
        d_cl_z = rbf_z(current_centerline[:,0], current_centerline[:,1], current_centerline[:,2])
        
        d_m_x = rbf_x(current_mesh_verts[:,0], current_mesh_verts[:,1], current_mesh_verts[:,2])
        d_m_y = rbf_y(current_mesh_verts[:,0], current_mesh_verts[:,1], current_mesh_verts[:,2])
        d_m_z = rbf_z(current_mesh_verts[:,0], current_mesh_verts[:,1], current_mesh_verts[:,2])
        
        dense_disp_cl = np.stack([d_cl_x, d_cl_y, d_cl_z], axis=1)
        dense_disp_m = np.stack([d_m_x, d_m_y, d_m_z], axis=1)

        # 3. Gaussian Kinks (Fine Detail / Sharp Bends)
        if np.random.rand() < CONFIG["Gaussian_Kink_Prob"]:
            kink_force = np.random.uniform(-0.5, 0.5, size=3) # Small sharp force
            
            # Apply near tip
            dists_cl_kink = np.linalg.norm(current_centerline - tip_pos, axis=1)
            w_cl_kink = gaussian_kernel(dists_cl_kink, sigma=3.0)
            
            dists_m_kink = np.linalg.norm(current_mesh_verts - tip_pos, axis=1)
            w_m_kink = gaussian_kernel(dists_m_kink, sigma=3.0)
            
            dense_disp_cl += np.outer(w_cl_kink, kink_force)
            dense_disp_m += np.outer(w_m_kink, kink_force)

        # --- D. Apply & Relax ---
        
        # Apply
        current_centerline += dense_disp_cl
        current_mesh_verts += dense_disp_m
        
        # Viscoelastic Relaxation (Pull back to Rest)
        relax_vec_cl = centerline - current_centerline
        relax_vec_m = mesh.vertices - current_mesh_verts
        
        current_centerline += relax_vec_cl * (1.0 - CONFIG["RELAXATION_FACTOR"])
        current_mesh_verts += relax_vec_m * (1.0 - CONFIG["RELAXATION_FACTOR"])

        # --- E. Save Data ---
        
        # Save Geometry
        temp_mesh = mesh.copy()
        temp_mesh.vertices = current_mesh_verts
        
        file_prefix = f"branch_{branch_id:02d}_pass_{pass_id:02d}_frame_{frame_count:03d}"
        
        # Save Targets
        np.save(os.path.join(output_dir, f"{file_prefix}_centerline.npy"), current_centerline)
        temp_mesh.export(os.path.join(output_dir, f"{file_prefix}_mesh.stl"))
        
        # Save Training Inputs (Sparse Parameters)
        # Note: We save the *accumulated* displacement from rest, not just the incremental delta
        total_sparse_disp = current_centerline[cp_indices] - control_points_rest
        np.save(os.path.join(output_dir, f"{file_prefix}_sparse_disp.npy"), total_sparse_disp)
        np.save(os.path.join(output_dir, f"{file_prefix}_cp_indices.npy"), cp_indices)

        # --- F. Stats & Viz ---
        new_edges = temp_mesh.edges_unique_length
        ratios = new_edges / (original_edges + 1e-6)
        max_strain = np.max(ratios)
        
        pass_stats["frames"].append({
            "frame": frame_count,
            "max_strain": float(max_strain),
            "force_vec": current_force_vec.tolist()
        })
        
        # Visualization Plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(centerline[:,0], centerline[:,1], centerline[:,2], c='gray', alpha=0.2, label="Rest")
        ax.plot(current_centerline[:,0], current_centerline[:,1], current_centerline[:,2], c='red', linewidth=2, label="Hybrid Deform")
        ax.scatter([tip_pos[0]], [tip_pos[1]], [tip_pos[2]], c='orange', s=100, marker='*')
        ax.quiver(tip_pos[0], tip_pos[1], tip_pos[2], 
                  current_force_vec[0], current_force_vec[1], current_force_vec[2], 
                  color='green', length=2.0)

        # Camera centering
        mid = np.mean(centerline, axis=0)
        rng = 30
        ax.set_xlim(mid[0]-rng, mid[0]+rng); ax.set_ylim(mid[1]-rng, mid[1]+rng); ax.set_zlim(mid[2]-rng, mid[2]+rng)
        ax.set_title(f"Hybrid Sim | B{branch_id} P{pass_id} F{frame_count}")
        
        frame_path = os.path.join(viz_dir, f"viz_{frame_count:03d}.png")
        plt.savefig(frame_path)
        plt.close()
        gif_frames.append(imageio.imread(frame_path))
        
        frame_count += 1
        if frame_count >= CONFIG["FRAMES_PER_PASS"]: break

    # Save GIF
    gif_path = os.path.join(output_dir, f"hybrid_branch_{branch_id}_pass_{pass_id}.gif")
    imageio.mimsave(gif_path, gif_frames, fps=10)
    shutil.rmtree(viz_dir)
    
    report_data["simulations"].append(pass_stats)

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    input_stl = "vessel.stl" 
    output_dir = "output_sequential"
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # --- LOAD MESH ---
    if not os.path.exists(input_stl):
        print("[INFO] 'vessel.stl' not found. Generating dummy tube...")
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

    # --- SKELETONIZE ---
    centerline = extract_centerline(mesh)
    
    # --- RUN SIMULATION ---
    report_data = {
        "config": CONFIG,
        "mesh_info": {"verts": len(mesh.vertices), "faces": len(mesh.faces)},
        "simulations": []
    }
    
    print(f"Starting Simulation of {CONFIG['NUM_BRANCHES']} Branches x {CONFIG['PASSES_PER_BRANCH']} Passes...")
    
    # --- OUTER LOOP: BRANCHES (Anatomy) ---
    for b in range(CONFIG["NUM_BRANCHES"]):
        
        # CRITICAL FIX: SELECT ANATOMY HERE (Once per Branch ID)
        print(f"\n--- Selecting Anatomy for Branch {b} ---")
        path_indices = get_path_segment(centerline, length_window=60)
        
        # --- INNER LOOP: PASSES (Variations) ---
        for p in range(CONFIG["PASSES_PER_BRANCH"]):
            # Pass the LOCKED 'path_indices' to the simulator
            apply_cumulative_hybrid_deformation(mesh, centerline, output_dir, b, p, report_data, path_indices)

    # --- SAVE REPORT ---
    with open(os.path.join(output_dir, "simulation_report.json"), 'w') as f:
        json.dump(report_data, f, indent=2)
        
    print("\n" + "="*50)
    print("SIMULATION COMPLETE")
    print(f"Data saved to: {output_dir}")
    print("="*50)