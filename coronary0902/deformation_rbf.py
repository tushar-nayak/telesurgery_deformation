import numpy as np
import trimesh
from skimage.morphology import skeletonize
import os
import json
from scipy.interpolate import Rbf

# --- Physics Kernels ---
def thin_plate_spline(r):
    """
    Slide 10: phi(r) = r^2 * log(r)
    Robust interpolation that minimizes bending energy.
    """
    if r == 0: return 0
    return (r**2) * np.log(r + 1e-6)

def gaussian_kernel(r, sigma=5.0):
    return np.exp(-r**2 / (2 * sigma**2))

# --- The New Deformation Logic ---
def apply_hybrid_deformation(mesh, centerline, num_control_points=10):
    """
    Deforms mesh using BOTH Sparse RBF Control Points AND Local Gaussian bumps.
    """
    N = len(centerline)
    mesh_verts = mesh.vertices.copy()
    cl_points = centerline.copy()
    
    # 1. Select Sparse Control Points (Equidistant)
    # We pick indices to define the 'structure'
    indices = np.linspace(0, N-1, num_control_points, dtype=int)
    control_points_orig = cl_points[indices]
    
    # 2. Generate Random Displacements for Control Points
    # Create random force vectors
    sparse_displacements = np.random.uniform(-5, 5, size=control_points_orig.shape)
    
    # 3. RBF Interpolation (The "Global" Deformation)
    # We fit an RBF function: f(original_pos) -> displacement
    # Using 'thin_plate' matches your presentation
    rbf_x = Rbf(control_points_orig[:,0], control_points_orig[:,1], control_points_orig[:,2], 
                sparse_displacements[:,0], function='thin_plate')
    rbf_y = Rbf(control_points_orig[:,0], control_points_orig[:,1], control_points_orig[:,2], 
                sparse_displacements[:,1], function='thin_plate')
    rbf_z = Rbf(control_points_orig[:,0], control_points_orig[:,1], control_points_orig[:,2], 
                sparse_displacements[:,2], function='thin_plate')
    
    # Apply RBF to ALL Centerline points
    delta_x = rbf_x(cl_points[:,0], cl_points[:,1], cl_points[:,2])
    delta_y = rbf_y(cl_points[:,0], cl_points[:,1], cl_points[:,2])
    delta_z = rbf_z(cl_points[:,0], cl_points[:,1], cl_points[:,2])
    dense_displacement_rbf = np.stack([delta_x, delta_y, delta_z], axis=1)
    
    # Apply RBF to ALL Mesh Vertices
    m_dx = rbf_x(mesh_verts[:,0], mesh_verts[:,1], mesh_verts[:,2])
    m_dy = rbf_y(mesh_verts[:,0], mesh_verts[:,1], mesh_verts[:,2])
    m_dz = rbf_z(mesh_verts[:,0], mesh_verts[:,1], mesh_verts[:,2])
    mesh_disp = np.stack([m_dx, m_dy, m_dz], axis=1)
    
    # 4. (Optional) Add Localized Gaussian Bump
    # This adds "fine detail" or "kinks" that RBF might smooth out too much
    if np.random.rand() > 0.5:
        # Pick random node
        idx = np.random.randint(0, N)
        center = cl_points[idx]
        force = np.random.uniform(-3, 3, size=3)
        
        # Calc weights
        dists_cl = np.linalg.norm(cl_points - center, axis=1)
        w_cl = gaussian_kernel(dists_cl, sigma=4.0)
        gaussian_disp_cl = np.outer(w_cl, force)
        
        dists_mesh = np.linalg.norm(mesh_verts - center, axis=1)
        w_mesh = gaussian_kernel(dists_mesh, sigma=4.0)
        gaussian_disp_mesh = np.outer(w_mesh, force)
        
        # Add to total
        dense_displacement_rbf += gaussian_disp_cl
        mesh_disp += gaussian_disp_mesh

    # Apply final deformation
    final_cl = cl_points + dense_displacement_rbf
    mesh.vertices += mesh_disp
    
    return mesh, final_cl, sparse_displacements, indices

# --- Main Execution ---
if __name__ == "__main__":
    # Settings
    input_stl = "vessel.stl" # Ensure this exists!
    output_dir = "output_meshes_rbf"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    print("Loading base mesh...")
    mesh = trimesh.load(input_stl)
    # Simple centerline extraction (replace with your robust one if needed)
    # For demo, taking a subset of vertices as pseudo-centerline to run fast
    # Ideally, load your pre-calculated 'base_centerline.npy' here
    # cl_points = ... 
    
    # Placeholder base centerline (replace with actual logic)
    # We will assume you generate valid base_cl
    from skimage.morphology import skeletonize
    voxel_grid = mesh.voxelized(pitch=0.5).fill()
    skel = skeletonize(voxel_grid.matrix)
    cl_points = voxel_grid.indices_to_points(np.argwhere(skel))
    
    # Sort points roughly
    # (Simple sort by Z for demo; use your advanced sorter in prod)
    cl_points = cl_points[np.argsort(cl_points[:, 2])] 

    print(f"Generating samples...")
    metadata = []
    
    for i in range(20): # Generate 20 samples
        new_mesh = mesh.copy()
        
        # APPLY NEW LOGIC
        def_mesh, def_cl, sparse_vecs, cp_indices = apply_hybrid_deformation(new_mesh, cl_points)
        
        base_name = f"sample_{i}"
        
        # Save Geometry
        np.save(f"{output_dir}/{base_name}_centerline.npy", def_cl)
        def_mesh.export(f"{output_dir}/{base_name}_mesh.stl")
        
        # Save CONTROL PARAMETERS (Labels for the network)
        np.save(f"{output_dir}/{base_name}_sparse_disp.npy", sparse_vecs)
        np.save(f"{output_dir}/{base_name}_cp_indices.npy", cp_indices)
        
        print(f"  Saved {base_name}")

    print("Done! Now run generate_dataset_final.py (modified to read this folder) to create images.")