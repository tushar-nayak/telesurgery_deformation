import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Synthetic Centerline Generator ---
def generate_synthetic_vessel(num_points=100, randomness=0.5):
    """
    Generates a random 3D smooth curve representing a vessel centerline.
    Uses a simple random walk smoothed by a cumulative sum.
    """
    t = np.linspace(0, 10, num_points)
    
    # Create a smooth, curving shape (like a neuro vessel)
    # We mix sine waves with some random noise
    x = np.sin(t) * 5 + np.random.normal(0, randomness, num_points)
    y = np.cos(t) * 5 + np.random.normal(0, randomness, num_points)
    z = t * 2 + np.random.normal(0, randomness, num_points)

    # Stack into (N, 3) array
    centerline = np.column_stack((x, y, z))
    
    # Smooth it out to look like a vessel (moving average)
    # Real vessels aren't jagged
    window_size = 5
    centerline_smooth = np.zeros_like(centerline)
    for i in range(3):
        centerline_smooth[:, i] = np.convolve(centerline[:, i], np.ones(window_size)/window_size, mode='same')
    
    # Trim artifacts from convolution
    return centerline_smooth[window_size:-window_size]

# --- 2. RBF Deformation Logic (TPS) ---
def thin_plate_spline_kernel(r):
    """
    TPS Kernel: phi(r) = r^2 * ln(r)
    Handles the case where r=0 to avoid log(0) errors.
    """
    res = np.zeros_like(r)
    # Only calculate log for r > 0
    mask = r > 1e-9
    res[mask] = (r[mask]**2) * np.log(r[mask])
    return res

def deform_vessel_rbf(centerline, num_control_points=1, force_scale=2.0):
    """
    Simulates guidewire deformation.
    1. Picks random control points on the vessel.
    2. Applies a random 'force' vector to them.
    3. Propagates this deformation to all other points using TPS-RBF.
    """
    N = len(centerline)
    
    # 1. Pick random indices to be our "contact points"
    control_indices = np.random.choice(N, num_control_points, replace=False)
    control_points = centerline[control_indices]
    
    # 2. Generate random force vectors (weights) for these points
    # Shape: (num_control_points, 3)
    forces = np.random.uniform(-1, 1, size=(num_control_points, 3))
    
    # Normalize and scale forces
    forces = (forces / np.linalg.norm(forces, axis=1, keepdims=True)) * force_scale
    
    # 3. Apply RBF to calculate displacement for EVERY point on the centerline
    # Distance matrix between all centerline points and control points
    # shape: (N, num_control_points)
    dists = np.linalg.norm(centerline[:, np.newaxis, :] - control_points[np.newaxis, :, :], axis=2)
    
    # Apply Kernel
    weights = thin_plate_spline_kernel(dists) # Shape (N, num_control_points)
    
    # Sum the weighted forces: Displacement = Sum( Weight_i * Force_i )
    # We need to map the scalar weights to the 3D force vectors
    # Resulting displacement shape: (N, 3)
    
    # NOTE: TPS is unbounded, so we often normalize or use a Gaussian if we want local only.
    # For TPS, deformations are global (infinite support).
    # To keep it realistic, we might dampen it, but here is raw TPS as requested:
    displacements = np.dot(weights, forces)
    
    # Normalize displacements slightly so the vessel doesn't explode
    # (TPS values can get very large for large distances)
    max_disp = np.max(np.linalg.norm(displacements, axis=1))
    if max_disp > 0:
        displacements = displacements / max_disp * force_scale
    
    deformed_centerline = centerline + displacements
    
    return deformed_centerline, control_indices, forces

# --- 3. Differential Projector (Simple Perspective) ---
def project_to_2d(points_3d, view_matrix=None, focal_length=1000):
    """
    Projects 3D points to 2D plane.
    Simulates a C-arm X-ray.
    """
    # If no view matrix, assume looking down Z-axis
    if view_matrix is None:
        view_matrix = np.eye(4)
        # Move camera back so objects are in front
        view_matrix[2, 3] = -50 
        
    num_points = len(points_3d)
    
    # Add homogeneous coordinate (w=1)
    points_hom = np.hstack((points_3d, np.ones((num_points, 1))))
    
    # Apply View Matrix (World -> Camera Space)
    points_cam = points_hom.dot(view_matrix.T)
    
    # Perspective Divide (x/z, y/z)
    # Avoid division by zero
    z = points_cam[:, 2] + 1e-5
    
    u = focal_length * (points_cam[:, 0] / z)
    v = focal_length * (points_cam[:, 1] / z)
    
    return np.column_stack((u, v))

# --- MAIN EXECUTION ---

# 1. Generate Data
original_cl = generate_synthetic_vessel(num_points=100)

# 2. Deform
deformed_cl, c_indices, c_forces = deform_vessel_rbf(
    original_cl, 
    num_control_points=2,   # Simulating 2 contact points
    force_scale=3.0         # Magnitude of deformation
)

# 3. Project
projected_2d = project_to_2d(deformed_cl)

# --- VISUALIZATION ---
fig = plt.figure(figsize=(15, 5))

# Plot 1: 3D Comparison
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(original_cl[:,0], original_cl[:,1], original_cl[:,2], 'b--', label='Original', alpha=0.5)
ax1.plot(deformed_cl[:,0], deformed_cl[:,1], deformed_cl[:,2], 'r-', label='Deformed', linewidth=2)
# Show control point force vectors
cp = original_cl[c_indices]
ax1.quiver(cp[:,0], cp[:,1], cp[:,2], c_forces[:,0], c_forces[:,1], c_forces[:,2], color='k', length=2)
ax1.set_title("3D Deformation (Red)")
ax1.legend()

# Plot 2: The "Input" 2D X-ray (Projection)
ax2 = fig.add_subplot(132)
ax2.plot(projected_2d[:,0], projected_2d[:,1], 'k-', linewidth=3, alpha=0.8)
ax2.set_title("Simulated 2D X-ray")
ax2.set_aspect('equal')
ax2.invert_yaxis() # Images have (0,0) at top-left
ax2.grid(True)

plt.tight_layout()
plt.show()

# Save Data Pair Example
data_pair = {
    "original_centerline": original_cl,
    "deformed_centerline": deformed_cl,
    "deformation_params": {
        "indices": c_indices,
        "forces": c_forces
    },
    "image_2d": projected_2d
}

print(f"Data generated. Original shape: {original_cl.shape}, Deformed shape: {deformed_cl.shape}")