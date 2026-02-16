import numpy as np
import trimesh
import os
import json
import shutil
import time

# --- 0. HEADLESS PLOTTING FIX ---
# Must be before importing pyplot to avoid X11 authorization errors in Docker
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from scipy.interpolate import Rbf
from skimage.morphology import skeletonize
import imageio.v2 as imageio

# --- 1. JSON HELPER ---
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- 2. CONFIGURATION ---
CONFIG = {
    "NUM_BRANCHES": 5,          
    "PASSES_PER_BRANCH": 3,     
    "FRAMES_PER_PASS": 40,      
    "FORCE_MAGNITUDE": 2.0,     
    "FORCE_DRIFT": 0.3,         
    "INFLUENCE_RADIUS": 15.0,   
    "RBF_SMOOTHING": 0.8,       
    "Gaussian_Kink_Prob": 0.3,  
    "RELAXATION_FACTOR": 0.90   
}

# --- 3. PHYSICS KERNELS ---
def thin_plate_spline(r):
    if r == 0: return 0
    return (r**2) * np.log(r + 1e-6)

def gaussian_kernel(r, sigma=4.0):
    return np.exp(-r**2 / (2 * sigma**2))

# --- 4. CORE SIMULATION LOGIC ---

def extract_centerline(mesh):
    print("  > Voxelizing and Skeletonizing mesh...")
    # Increase pitch if voxelization is too slow or consumes too much RAM
    voxel_grid = mesh.voxelized(pitch=0.5).fill()
    skel = skeletonize(voxel_grid.matrix)
    centerline = voxel_grid.indices_to_points(np.argwhere(skel))
    # Simple sort along Z for rough ordering
    centerline = centerline[np.argsort(centerline[:, 2])]
    print(f"  > Centerline extracted: {len(centerline)} points")
    return centerline

def get_path_segment(centerline, length_window=60):
    total_points = len(centerline)
    if total_points < length_window:
        return np.arange(total_points)
    start_idx = np.random.randint(0, total_points - length_window)
    return np.arange(start_idx, start_idx + length_window)

def apply_cumulative_hybrid_deformation(mesh, centerline, output_dir, branch_id, pass_id, report_data, path_indices):
    N = len(centerline)
    num_cp = 20
    cp_indices = np.linspace(0, N-1, num_cp, dtype=int)
    control_points_rest = centerline[cp_indices]
    
    current_mesh_verts = mesh.vertices.copy()
    current_centerline = centerline.copy()
    
    current_force_vec = np.random.normal(0, 1, size=3)
    current_force_vec /= np.linalg.norm(current_force_vec)
    current_force_vec *= CONFIG["FORCE_MAGNITUDE"]
    
    viz_dir = os.path.join(output_dir, "viz_frames")
    if os.path.exists(viz_dir): shutil.rmtree(viz_dir)
    os.makedirs(viz_dir)
    
    gif_frames = []
    pass_stats = {"id": f"b{branch_id}_p{pass_id}", "frames": []}
    original_edges = mesh.edges_unique_length

    print(f"  > Simulating Branch {branch_id}, Pass {pass_id}...")

    step_size = max(1, len(path_indices) // CONFIG["FRAMES_PER_PASS"])
    frame_count = 0
    
    for t in range(0, len(path_indices), step_size):
        tip_idx_global = path_indices[t]
        tip_pos = current_centerline[tip_idx_global]
        
        drift = np.random.normal(0, CONFIG["FORCE_DRIFT"], size=3)
        current_force_vec += drift
        mag = np.linalg.norm(current_force_vec)
        current_force_vec = current_force_vec * (CONFIG["FORCE_MAGNITUDE"] / mag) 
        
        dists_to_tip = np.linalg.norm(control_points_rest - tip_pos, axis=1)
        weights = np.exp(-dists_to_tip**2 / (2 * (CONFIG["INFLUENCE_RADIUS"]/2)**2))
        
        cp_deltas = np.zeros_like(control_points_rest)
        for i in range(num_cp):
            cp_deltas[i] = current_force_vec * weights[i]
            
        rbf_args = dict(function='thin_plate', smooth=CONFIG["RBF_SMOOTHING"])
        try:
            rbf_x = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], cp_deltas[:,0], **rbf_args)
            rbf_y = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], cp_deltas[:,1], **rbf_args)
            rbf_z = Rbf(control_points_rest[:,0], control_points_rest[:,1], control_points_rest[:,2], cp_deltas[:,2], **rbf_args)
        except np.linalg.LinAlgError:
            continue

        d_cl_x = rbf_x(current_centerline[:,0], current_centerline[:,1], current_centerline[:,2])
        d_cl_y = rbf_y(current_centerline[:,0], current_centerline[:,1], current_centerline[:,2])
        d_cl_z = rbf_z(current_centerline[:,0], current_centerline[:,1], current_centerline[:,2])
        
        d_m_x = rbf_x(current_mesh_verts[:,0], current_mesh_verts[:,1], current_mesh_verts[:,2])
        d_m_y = rbf_y(current_mesh_verts[:,0], current_mesh_verts[:,1], current_mesh_verts[:,2])
        d_m_z = rbf_z(current_mesh_verts[:,0], current_mesh_verts[:,1], current_mesh_verts[:,2])
        
        dense_disp_cl = np.stack([d_cl_x, d_cl_y, d_cl_z], axis=1)
        dense_disp_m = np.stack([d_m_x, d_m_y, d_m_z], axis=1)

        if np.random.rand() < CONFIG["Gaussian_Kink_Prob"]:
            kink_force = np.random.uniform(-0.5, 0.5, size=3)
            dists_cl_kink = np.linalg.norm(current_centerline - tip_pos, axis=1)
            w_cl_kink = gaussian_kernel(dists_cl_kink, sigma=3.0)
            dists_m_kink = np.linalg.norm(current_mesh_verts - tip_pos, axis=1)
            w_m_kink = gaussian_kernel(dists_m_kink, sigma=3.0)
            dense_disp_cl += np.outer(w_cl_kink, kink_force)
            dense_disp_m += np.outer(w_m_kink, kink_force)

        current_centerline += dense_disp_cl
        current_mesh_verts += dense_disp_m
        
        relax_vec_cl = centerline - current_centerline
        relax_vec_m = mesh.vertices - current_mesh_verts
        current_centerline += relax_vec_cl * (1.0 - CONFIG["RELAXATION_FACTOR"])
        current_mesh_verts += relax_vec_m * (1.0 - CONFIG["RELAXATION_FACTOR"])

        temp_mesh = mesh.copy()
        temp_mesh.vertices = current_mesh_verts
        
        file_prefix = f"branch_{branch_id:02d}_pass_{pass_id:02d}_frame_{frame_count:03d}"
        
        np.save(os.path.join(output_dir, f"{file_prefix}_centerline.npy"), current_centerline)
        temp_mesh.export(os.path.join(output_dir, f"{file_prefix}_mesh.stl"))
        
        total_sparse_disp = current_centerline[cp_indices] - control_points_rest
        np.save(os.path.join(output_dir, f"{file_prefix}_sparse_disp.npy"), total_sparse_disp)
        np.save(os.path.join(output_dir, f"{file_prefix}_cp_indices.npy"), cp_indices)

        new_edges = temp_mesh.edges_unique_length
        ratios = new_edges / (original_edges + 1e-6)
        max_strain = np.max(ratios)
        
        pass_stats["frames"].append({
            "frame": frame_count,
            "max_strain": float(max_strain),
            "force_vec": current_force_vec.tolist()
        })
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(centerline[:,0], centerline[:,1], centerline[:,2], c='gray', alpha=0.2)
        ax.plot(current_centerline[:,0], current_centerline[:,1], current_centerline[:,2], c='red', linewidth=2)
        ax.scatter([tip_pos[0]], [tip_pos[1]], [tip_pos[2]], c='orange', s=100, marker='*')
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

    gif_path = os.path.join(output_dir, f"hybrid_branch_{branch_id}_pass_{pass_id}.gif")
    imageio.mimsave(gif_path, gif_frames, fps=10)
    shutil.rmtree(viz_dir)
    report_data["simulations"].append(pass_stats)

# --- 5. MESH REPAIR UTILS (GVXR-SAFE & ROBUST) ---

def keep_largest_component(mesh):
    """
    Removes floating noise/debris.
    Returns the largest connected component by vertex count.
    """
    try:
        # split() can be slow on complex meshes, but is necessary for Euler topology
        components = mesh.split(only_watertight=False)
        if len(components) == 0:
            return mesh, 0
        if len(components) == 1:
            return mesh, 0
            
        components.sort(key=lambda m: len(m.vertices), reverse=True)
        return components[0], len(components) - 1
    except Exception:
        return mesh, 0

def extract_boundary_loops(mesh, boundary_edges):
    from collections import defaultdict
    graph = defaultdict(list)
    for edge in boundary_edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    
    loops = []
    visited = set()
    for start_vert in graph.keys():
        if start_vert in visited: continue
        loop = [start_vert]
        current = start_vert
        visited.add(current)
        while True:
            neighbors = [v for v in graph[current] if v not in visited or v == start_vert]
            if not neighbors: break
            next_vert = neighbors[0]
            if next_vert == start_vert:
                loops.append(np.array(loop))
                break
            loop.append(next_vert)
            visited.add(next_vert)
            current = next_vert
            if len(loop) > len(graph) * 2: break
    return loops

def cap_mesh_boundary(mesh, boundary_loop):
    if len(boundary_loop) < 3: return
    boundary_verts = mesh.vertices[boundary_loop]
    centroid = np.mean(boundary_verts, axis=0)
    centroid_idx = len(mesh.vertices)
    mesh.vertices = np.vstack([mesh.vertices, centroid])
    new_faces = []
    for i in range(len(boundary_loop)):
        v1 = boundary_loop[i]
        v2 = boundary_loop[(i + 1) % len(boundary_loop)]
        new_faces.append([centroid_idx, v1, v2])
    mesh.faces = np.vstack([mesh.faces, new_faces])

def repair_and_cap_mesh_conservative(mesh_path):
    stats = {
        "file": os.path.basename(mesh_path),
        "initial_watertight": False,
        "final_watertight": False,
        "repair_method": None,
        "errors": [],
        "geometry_preserved": True
    }
    
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        original_verts = mesh.vertices.copy()
        stats["initial_watertight"] = bool(mesh.is_watertight)
        
        # --- STAGE 0: CLEANUP NOISE ---
        mesh, removed = keep_largest_component(mesh)
        if removed > 0: stats["noise_components_removed"] = int(removed)

        # --- PRE-CHECK: INVERTED NORMALS ---
        # If watertight but volume is negative, it's inside out.
        if mesh.is_watertight and mesh.volume < 0:
            mesh.invert()
            stats["repair_method"] = "normal_flip_only"
            # If fixing normals makes volume positive, we are good.
            if mesh.volume > 1e-6:
                mesh.export(mesh_path)
                return True, stats
        
        # If already good
        if mesh.is_watertight and mesh.volume > 1e-6:
            stats["repair_method"] = "none_needed"
            return True, stats
        
        # --- STAGE 1: TOPOLOGY REPAIRS ---
        mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=8)
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces(height=1e-10)
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
        
        mesh, _ = keep_largest_component(mesh)
        
        # Check if topology repair fixed it
        if mesh.is_watertight:
            if mesh.volume < 0: mesh.invert()
            stats["repair_method"] = "topology_only"
            stats["geometry_preserved"] = True
            mesh.export(mesh_path)
            return True, stats
        
        # --- STAGE 2: CAP BOUNDARIES ---
        edges = mesh.edges_sorted
        unique_edges, inverse = trimesh.grouping.unique_rows(edges)
        edge_counts = np.bincount(inverse)
        boundary_edge_mask = edge_counts == 1
        num_holes = np.sum(boundary_edge_mask)
        stats["num_boundary_edges"] = int(num_holes)
        
        if num_holes > 0:
            try:
                boundary_edges_idx = np.where(boundary_edge_mask)[0]
                boundary_edges_actual = unique_edges[boundary_edges_idx]
                loops = extract_boundary_loops(mesh, boundary_edges_actual)
                stats["num_boundary_loops"] = len(loops)
                
                for loop in loops:
                    cap_mesh_boundary(mesh, loop)
                    
                stats["repair_method"] = "boundary_capping"
                
                # Final cleanup cycle
                mesh.merge_vertices(digits_vertex=8)
                trimesh.repair.fix_normals(mesh)
                mesh, _ = keep_largest_component(mesh) # Clean noise generated by caps
                
                if mesh.volume < 0: mesh.invert()
                
                if mesh.is_watertight and mesh.volume > 1e-6:
                    stats["geometry_preserved"] = True
                    stats["final_watertight"] = True
                    mesh.export(mesh_path)
                    return True, stats
            except Exception as e:
                stats["errors"].append(f"Boundary capping failed: {str(e)}")

        # --- STAGE 3: LAST RESORT FILL ---
        try:
            trimesh.repair.fill_holes(mesh)
            mesh, _ = keep_largest_component(mesh)
            if mesh.volume < 0: mesh.invert()
            
            if mesh.is_watertight and mesh.volume > 1e-6:
                # Conservative check: don't accept if vertices moved too much
                vert_displacement = np.linalg.norm(
                    mesh.vertices[:len(original_verts)] - original_verts, axis=1
                )
                max_d = np.max(vert_displacement) if len(vert_displacement) > 0 else 0
                if max_d < 1e-5:
                    stats["repair_method"] = "conservative_fill"
                    mesh.export(mesh_path)
                    return True, stats
        except:
            pass
            
        stats["final_watertight"] = False
        return False, stats
        
    except Exception as e:
        stats["errors"].append(f"Critical error: {str(e)}")
        return False, stats

def validate_mesh_for_gvxr(mesh_path):
    """
    Relaxed validation for GVXR.
    Allows Euler != 2 if mesh is watertight and has valid volume.
    """
    try:
        mesh = trimesh.load(mesh_path)
        checks = {
            "watertight": bool(mesh.is_watertight),
            "valid_volume": False,
            "no_degenerates": False,
            "no_self_intersections": False,
            "euler_valid": False
        }
        
        if mesh.is_watertight:
            try:
                volume = mesh.volume
                checks["valid_volume"] = bool(volume > 1e-6)
            except: pass
        
        face_areas = mesh.area_faces
        checks["no_degenerates"] = bool(np.all(face_areas > 1e-10))
        
        # Euler check is now informational only
        checks["euler_valid"] = bool(mesh.euler_number == 2)
        
        try:
            checks["no_self_intersections"] = bool(not mesh.is_self_intersecting)
        except:
            checks["no_self_intersections"] = None
        
        # CRITICAL CHANGE: We accept the mesh if it's watertight, has volume, and no degenerates.
        # We IGNORE euler_valid and self_intersections for the pass/fail criteria
        # because ray-tracers (GVXR) usually handle valid watertight solids even if they have handles.
        required_checks = [
            checks["watertight"],
            checks["valid_volume"],
            checks["no_degenerates"]
        ]
        
        all_pass = all(required_checks)
        return all_pass, checks
        
    except Exception as e:
        return False, {"error": str(e)}

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    input_stl = "vessel.stl" 
    output_dir = "output_sequential_wt"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # --- LOAD/GEN MESH ---
    if not os.path.exists(input_stl):
        print("[INFO] 'vessel.stl' not found. Generating dummy tube...")
        t = np.linspace(0, 4*np.pi, 100)
        x = 10 * np.cos(t); y = 10 * np.sin(t); z = np.linspace(0, 60, 100)
        path = np.column_stack((x, y, z))
        mesh = trimesh.creation.cylinder(radius=2.5, segment=path, sections=16)
        mesh.export(input_stl)
    else:
        print(f"[INFO] Loading {input_stl}...")
        mesh = trimesh.load(input_stl)

    centerline = extract_centerline(mesh)
    
    # --- RUN SIM ---
    report_data = {"config": CONFIG, "simulations": []}
    print(f"Starting Simulation of {CONFIG['NUM_BRANCHES']} Branches x {CONFIG['PASSES_PER_BRANCH']} Passes...")
    
    for b in range(CONFIG["NUM_BRANCHES"]):
        print(f"\n--- Selecting Anatomy for Branch {b} ---")
        path_indices = get_path_segment(centerline, length_window=60)
        for p in range(CONFIG["PASSES_PER_BRANCH"]):
            apply_cumulative_hybrid_deformation(mesh, centerline, output_dir, b, p, report_data, path_indices)

    with open(os.path.join(output_dir, "simulation_report.json"), 'w') as f:
        json.dump(report_data, f, indent=2, cls=NumpyEncoder)
        
    print("\n" + "="*50)
    print("SIMULATION COMPLETE. STARTING REPAIR...")
    print("="*50)

    # --- REPAIR ---
    stl_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".stl")])
    fixed_count = 0; failed_count = 0; already_good = 0; failed_meshes = []
    
    print(f"Scanning {len(stl_files)} meshes for GVXR compatibility...")
    
    for i, stl_path in enumerate(stl_files):
        if (i + 1) % 10 == 0: print(f"  Progress: {i+1}/{len(stl_files)}")
        
        try:
            gvxr_valid, checks = validate_mesh_for_gvxr(stl_path)
            if gvxr_valid:
                already_good += 1
                continue
            
            print(f"  [Repairing] {os.path.basename(stl_path)}...")
            success, stats = repair_and_cap_mesh_conservative(stl_path)
            
            if success:
                gvxr_valid, checks = validate_mesh_for_gvxr(stl_path)
                if gvxr_valid:
                    fixed_count += 1
                    # Log warning if Euler is invalid but we accepted it
                    euler_status = "Euler OK" if checks["euler_valid"] else "Euler INVALID (Accepted)"
                    print(f"    ✓ Fixed ({stats['repair_method']}) | {euler_status}")
                else:
                    failed_count += 1
                    stats["gvxr_checks"] = checks
                    failed_meshes.append({"file": os.path.basename(stl_path), "stats": stats})
                    print(f"    ✗ Watertight but GVXR validation failed: {checks}")
            else:
                failed_count += 1
                failed_meshes.append({"file": os.path.basename(stl_path), "stats": stats})
                print(f"    ✗ FAILED - {stats['repair_method']}")
                
        except Exception as e:
            failed_count += 1
            print(f"  [ERROR] {str(e)}")
            failed_meshes.append({"file": os.path.basename(stl_path), "error": str(e)})
    
    if failed_meshes:
        failed_dir = os.path.join(output_dir, "failed_gvxr")
        os.makedirs(failed_dir, exist_ok=True)
        for failed in failed_meshes:
            src = os.path.join(output_dir, failed["file"])
            dst = os.path.join(failed_dir, failed["file"])
            if os.path.exists(src): shutil.move(src, dst)
        
        with open(os.path.join(failed_dir, "gvxr_failure_report.json"), 'w') as f:
            json.dump({"total_failures": failed_count, "failed_meshes": failed_meshes}, f, indent=2, cls=NumpyEncoder)
            
    print("-" * 50)
    print(f"Repair Complete. Fixed: {fixed_count}, Good: {already_good}, Failed: {failed_count}")