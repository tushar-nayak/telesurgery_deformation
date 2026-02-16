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
    output_dir = "output_sequential_fix"
    
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

# --- 5. APPENDED: MESH REPAIR UTILS ---


def repair_and_cap_mesh(mesh_path):
    """
    Loads an STL, repairs non-manifold geometry, ensures it is watertight,
    and overwrites the file ONLY if successful.
    
    Returns:
        (success: bool, stats: dict) - Repair status and diagnostic info
    """
    stats = {
        "file": os.path.basename(mesh_path),
        "initial_watertight": False,
        "final_watertight": False,
        "repair_method": None,
        "errors": []
    }
    
    try:
        mesh = trimesh.load(mesh_path)
        stats["initial_watertight"] = mesh.is_watertight
        stats["initial_verts"] = len(mesh.vertices)
        stats["initial_faces"] = len(mesh.faces)
        
        # 1. Fast Check: If already watertight, skip processing
        if mesh.is_watertight:
            stats["repair_method"] = "none_needed"
            return True, stats
        
        # Store original for fallback
        original_mesh = mesh.copy()
        
        # 2. Standard Repair Pipeline
        # Merge vertices with tolerance for numerical precision issues
        mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=4)
        
        # Remove bad geometry
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces(height=1e-6)
        mesh.remove_unreferenced_vertices()
        
        # Fix orientation
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
        
        # 3. Check manifold edges
        try:
            if len(mesh.edges) > 0:
                # Every edge should be shared by exactly 2 faces
                edges_count = len(mesh.edges)
                unique_edges = len(mesh.edges_unique)
                stats["edge_ratio"] = edges_count / (2 * unique_edges) if unique_edges > 0 else 0
        except Exception as e:
            stats["errors"].append(f"Edge check failed: {str(e)}")
        
        # 4. Iterative hole filling
        for attempt in range(3):
            try:
                if attempt == 0:
                    # First attempt: Standard hole filling
                    trimesh.repair.fill_holes(mesh)
                    stats["repair_method"] = "standard_fill"
                elif attempt == 1:
                    # Second attempt: Unlimited hole size
                    trimesh.repair.fill_holes(mesh, size_max=None)
                    stats["repair_method"] = "unlimited_fill"
                else:
                    # Third attempt: Re-merge and try again
                    mesh.merge_vertices(digits_vertex=3)  # More aggressive
                    trimesh.repair.fill_holes(mesh, size_max=None)
                    stats["repair_method"] = "aggressive_fill"
                
                if mesh.is_watertight:
                    break
                    
            except Exception as e:
                stats["errors"].append(f"Fill attempt {attempt+1}: {str(e)}")
        
        # 5. Nuclear Option: Voxel-based healing
        if not mesh.is_watertight:
            try:
                print(f"      [Voxel Heal] Attempting volumetric repair...")
                
                # Calculate appropriate voxel size (1-2% of largest dimension)
                voxel_pitch = mesh.extents.max() / 80
                
                # Voxelize → fill volume → reconstruct surface
                voxelized = mesh.voxelized(pitch=voxel_pitch)
                filled_voxels = voxelized.fill()
                mesh = filled_voxels.marching_cubes
                
                # Clean up marching cubes artifacts
                mesh.merge_vertices(digits_vertex=4)
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
                
                stats["repair_method"] = "voxel_reconstruction"
                
            except Exception as e:
                stats["errors"].append(f"Voxel heal failed: {str(e)}")
                # Restore original if voxel method crashes
                mesh = original_mesh
        
        # 6. Final Validation
        stats["final_watertight"] = mesh.is_watertight
        stats["final_verts"] = len(mesh.vertices)
        stats["final_faces"] = len(mesh.faces)
        
        # Calculate quality metrics if successful
        if mesh.is_watertight:
            try:
                stats["volume"] = float(mesh.volume)
                stats["surface_area"] = float(mesh.area)
                stats["euler_number"] = mesh.euler_number
            except Exception as e:
                stats["errors"].append(f"Quality metrics failed: {str(e)}")
        
        # 7. Save ONLY if watertight
        if mesh.is_watertight:
            mesh.export(mesh_path)
            return True, stats
        else:
            # Don't overwrite with broken mesh
            return False, stats
            
    except Exception as e:
        stats["errors"].append(f"Critical error: {str(e)}")
        return False, stats


# --- EXECUTE REPAIR PASS ---
if 'output_dir' in locals():
    print("\n" + "="*50)
    print("POST-PROCESSING: REPAIRING MESHES")
    print("="*50)
    
    stl_files = sorted([
        os.path.join(output_dir, f) 
        for f in os.listdir(output_dir) 
        if f.endswith(".stl")
    ])
    
    fixed_count = 0
    failed_count = 0
    already_good = 0
    failed_meshes = []
    
    print(f"Scanning {len(stl_files)} meshes for watertightness...")
    
    start_time = time.time()
    
    for i, stl_path in enumerate(stl_files):
        # Progress indicator
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(stl_files) - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{len(stl_files)} ({rate:.1f} mesh/s, ETA: {eta:.0f}s)")
        
        try:
            # Quick pre-check to avoid loading if already good
            m = trimesh.load(stl_path)
            if m.is_watertight:
                already_good += 1
                continue
            
            # Attempt repair
            print(f"  [Repairing] {os.path.basename(stl_path)}...")
            success, stats = repair_and_cap_mesh(stl_path)
            
            if success:
                fixed_count += 1
                print(f"    ✓ Fixed using: {stats['repair_method']}")
            else:
                failed_count += 1
                failed_meshes.append({
                    "file": os.path.basename(stl_path),
                    "stats": stats
                })
                print(f"    ✗ FAILED - Mesh remains non-watertight")
                if stats["errors"]:
                    print(f"      Errors: {stats['errors']}")
                
        except Exception as e:
            failed_count += 1
            print(f"  [ERROR] Processing {os.path.basename(stl_path)}: {str(e)}")
            failed_meshes.append({
                "file": os.path.basename(stl_path),
                "stats": {"errors": [str(e)]}
            })
    
    total_time = time.time() - start_time
    
    # Summary Report
    print("-" * 50)
    print(f"Repair Complete in {total_time:.1f}s")
    print(f"  - Already Watertight: {already_good}")
    print(f"  - Successfully Repaired: {fixed_count}")
    print(f"  - Failed to Repair: {failed_count}")
    print(f"  - Total Processed: {len(stl_files)}")
    print(f"  - Success Rate: {100 * (already_good + fixed_count) / len(stl_files):.1f}%")
    
    # Handle failed meshes
    if failed_meshes:
        print("\n" + "="*50)
        print("HANDLING FAILED MESHES")
        print("="*50)
        
        # Move failed meshes to separate directory
        failed_dir = os.path.join(output_dir, "failed_repairs")
        os.makedirs(failed_dir, exist_ok=True)
        
        for failed in failed_meshes:
            src = os.path.join(output_dir, failed["file"])
            dst = os.path.join(failed_dir, failed["file"])
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"  Moved: {failed['file']}")
        
        # Save detailed failure report
        failure_report = {
            "total_failures": failed_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "failed_meshes": failed_meshes
        }
        
        report_path = os.path.join(failed_dir, "failure_report.json")
        with open(report_path, 'w') as f:
            json.dump(failure_report, f, indent=2)
        
        print(f"\n  Failed meshes moved to: {failed_dir}")
        print(f"  Failure report saved to: {report_path}")
        
        # Update main simulation report to exclude failed meshes
        main_report_path = os.path.join(output_dir, "simulation_report.json")
        if os.path.exists(main_report_path):
            with open(main_report_path, 'r') as f:
                sim_report = json.load(f)
            
            sim_report["mesh_repair_summary"] = {
                "total_meshes": len(stl_files),
                "successful": already_good + fixed_count,
                "failed": failed_count,
                "failed_moved_to": "failed_repairs/"
            }
            
            with open(main_report_path, 'w') as f:
                json.dump(sim_report, f, indent=2)
            
            print(f"  Updated simulation report with repair stats")
    
    print("=" * 50)
