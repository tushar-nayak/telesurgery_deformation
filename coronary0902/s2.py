import numpy as np
import trimesh
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Load STL & Extract Centerline (FIXED) ---
def load_and_skeletonize(stl_path, voxel_pitch=0.5):
    """
    1. Loads an STL.
    2. Voxelizes it (turns it into a 3D block grid).
    3. Runs skeletonization (medial axis transform) to get the centerline.
    """
    print(f"Loading {stl_path}...")
    # Load the mesh
    mesh = trimesh.load_mesh(stl_path)
    
    # Voxelize the mesh
    # .fill() ensures the grid is dense (not sparse) so we can get a matrix
    # pitch = size of each voxel in mm (lower = more precise but slower)
    voxel_grid = mesh.voxelized(pitch=voxel_pitch).fill()
    
    # Extract the boolean matrix (True = vessel, False = empty)
    matrix = voxel_grid.matrix
    
    # Perform Skeletonization (Morphological thinning)
    print("Skeletonizing...")
    skeleton = skeletonize(matrix)
    
    # Get the indices of the skeletonized pixels (N, 3)
    indices = np.argwhere(skeleton)
    
    # Convert indices back to real-world 3D coordinates
    # Trimesh handles the origin/transform automatically here
    centerline_points = voxel_grid.indices_to_points(indices)

    # Sort points to make them a continuous line
    # (Simple nearest neighbor greedy sort to organize the point cloud into a line)
    if len(centerline_points) > 0:
        sorted_indices = [0]
        remaining = list(range(1, len(centerline_points)))
        
        while remaining:
            last_pt = centerline_points[sorted_indices[-1]]
            # Find closest remaining point
            dists = np.linalg.norm(centerline_points[remaining] - last_pt, axis=1)
            nearest = np.argmin(dists)
            sorted_indices.append(remaining[nearest])
            remaining.pop(nearest)
        
        centerline_points = centerline_points[sorted_indices]
        
    return mesh, centerline_points

# --- 2. RBF Deformation Logic (TPS) ---
def thin_plate_spline_kernel(r):
    res = np.zeros_like(r)
    # Avoid log(0)
    mask = r > 1e-9
    res[mask] = (r[mask]**2) * np.log(r[mask])
    return res

def deform_mesh_and_centerline(mesh, centerline, num_control_points=2, force_scale=5.0):
    """
    Applies the deformation to BOTH the mesh vertices and the centerline.
    This ensures the visual mesh and the physics graph stay in sync.
    """
    print("Applying deformation...")
    
    # 1. Pick control points from the CENTERLINE (guidewire pushes from inside)
    # Ensure we don't pick more points than exist
    actual_cp_count = min(num_control_points, len(centerline))
    c_indices = np.random.choice(len(centerline), actual_cp_count, replace=False)
    control_points = centerline[c_indices]
    
    # 2. Generate random forces
    forces = np.random.uniform(-1, 1, size=(actual_cp_count, 3))
    # Normalize and scale
    forces = (forces / np.linalg.norm(forces, axis=1, keepdims=True)) * force_scale
    
    # --- A. Deform the Mesh Vertices ---
    # Calc distance from every mesh vertex to the control points
    dists_mesh = np.linalg.norm(mesh.vertices[:, np.newaxis, :] - control_points[np.newaxis, :, :], axis=2)
    weights_mesh = thin_plate_spline_kernel(dists_mesh)
    
    # Calculate displacement for mesh
    disp_mesh = np.dot(weights_mesh, forces)
    
    # Normalize displacement to prevent explosion (TPS is unbounded)
    max_disp = np.max(np.linalg.norm(disp_mesh, axis=1))
    if max_disp > 0:
        disp_mesh = disp_mesh / max_disp * force_scale
        
    new_vertices = mesh.vertices + disp_mesh
    
    # Create new mesh object
    deformed_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)
    
    # --- B. Deform the Centerline ---
    # We must apply the EXACT same math to the centerline points
    dists_cl = np.linalg.norm(centerline[:, np.newaxis, :] - control_points[np.newaxis, :, :], axis=2)
    weights_cl = thin_plate_spline_kernel(dists_cl)
    disp_cl = np.dot(weights_cl, forces)
    
    if max_disp > 0:
        disp_cl = disp_cl / max_disp * force_scale
        
    deformed_centerline = centerline + disp_cl
    
    return deformed_mesh, deformed_centerline

# --- Main Example Usage ---
if __name__ == "__main__":
    # 1. Create a dummy STL for demonstration if 'bv.stl' doesn't exist
    # If you have your own 'bv.stl', you can comment out these two lines
    print("Creating dummy vessel.stl...")
    sphere = trimesh.creation.cylinder(radius=2, height=30, sections=20)
    sphere.export('bv.stl')

    # 2. Load and Extract
    # Note: voxel_pitch=0.5 is a good balance. Lower (0.1) is smoother but slower.
    try:
        original_mesh, original_cl = load_and_skeletonize('bv.stl', voxel_pitch=0.5)
        print(f"Centerline extracted with {len(original_cl)} points.")

        # 3. Deform
        deformed_mesh, deformed_cl = deform_mesh_and_centerline(original_mesh, original_cl)

        # 4. Visualize
        print("Visualizing...")
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection='3d')
        
        # Plot Original
        ax.scatter(original_cl[:,0], original_cl[:,1], original_cl[:,2], c='blue', label='Original', s=1)
        
        # Plot Deformed
        ax.scatter(deformed_cl[:,0], deformed_cl[:,1], deformed_cl[:,2], c='red', label='Deformed', s=1)
        
        ax.legend()
        ax.set_title("Centerline Extraction & Deformation")
        
        # Show plot
        plt.show()

        # Optional: Show the full 3D mesh (opens a separate window)
        # deformed_mesh.show()
        
    except Exception as e:
        print(f"An error occurred: {e}")