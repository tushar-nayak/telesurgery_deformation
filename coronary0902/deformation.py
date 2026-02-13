import numpy as np
import trimesh
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- 1. Load STL & Extract Centerline ---
def load_and_skeletonize(stl_path, voxel_pitch=0.5):
    """
    1. Loads an STL.
    2. Voxelizes it (turns it into a 3D block grid).
    3. Runs skeletonization (medial axis transform) to get the centerline.
    """
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"Could not find file: {stl_path}. Please check the filename.")

    print(f"Loading {stl_path}...")
    mesh = trimesh.load_mesh(stl_path)
    
    # Voxelize the mesh
    # .fill() ensures the grid is dense (not sparse) so we can get a matrix
    print("Voxelizing...")
    voxel_grid = mesh.voxelized(pitch=voxel_pitch).fill()
    
    # Extract the boolean matrix (True = vessel, False = empty)
    matrix = voxel_grid.matrix
    
    # Perform Skeletonization (Morphological thinning)
    print("Skeletonizing (this may take a moment)...")
    skeleton = skeletonize(matrix)
    
    # Get the indices of the skeletonized pixels (N, 3)
    indices = np.argwhere(skeleton)
    
    # Convert indices back to real-world 3D coordinates
    # Trimesh handles the origin/transform automatically here
    centerline_points = voxel_grid.indices_to_points(indices)

    # Sort points to make them a continuous line
    # (Simple nearest neighbor greedy sort)
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

# --- 2. RBF Deformation Logic (Physics) ---
def thin_plate_spline_kernel(r):
    res = np.zeros_like(r)
    # Avoid log(0) error
    mask = r > 1e-9
    res[mask] = (r[mask]**2) * np.log(r[mask])
    return res

def deform_mesh_and_centerline(mesh, centerline, num_control_points=2, force_scale=5.0):
    """
    Applies the deformation to BOTH the mesh vertices and the centerline.
    This ensures the visual mesh and the physics graph stay in sync.
    """
    print("Applying deformation physics...")
    
    # 1. Pick control points from the CENTERLINE (guidewire pushes from inside)
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

# --- Main Execution Block ---
if __name__ == "__main__":
    # !!! CHANGE THIS FILENAME TO MATCH YOUR FILE !!!
    stl_filename = "vessel.stl" 
    
    # Check if user forgot to change the name
    if stl_filename == "your_vessel.stl" and not os.path.exists(stl_filename):
        print(f"ERROR: You need to rename '{stl_filename}' in the code to your actual file name.")
    else:
        try:
            # 1. Load and Extract Centerline
            # voxel_pitch=0.5 is standard. Use 0.2 for higher precision (slower).
            original_mesh, original_cl = load_and_skeletonize(stl_filename, voxel_pitch=0.5)
            print(f"Success! Centerline extracted with {len(original_cl)} points.")

            # 2. Apply Simulated Deformation
            deformed_mesh, deformed_cl = deform_mesh_and_centerline(original_mesh, original_cl)

            # 3. Visualize Results
            print("Opening visualization window...")
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(121, projection='3d')
            
            # Plot Original (Blue)
            ax.scatter(original_cl[:,0], original_cl[:,1], original_cl[:,2], c='blue', label='Original', s=1)
            
            # Plot Deformed (Red)
            ax.scatter(deformed_cl[:,0], deformed_cl[:,1], deformed_cl[:,2], c='red', label='Deformed', s=1)
            
            ax.legend()
            ax.set_title("Vessel Deformation")
            plt.show()

            # Optional: Save the deformed mesh to check in other software
            deformed_mesh.export('deformed_output.stl')
            print("Saved deformed_output.stl")
            
        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")