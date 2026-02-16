from xray_depth_generator import XRayDepthGenerator
from itertools import permutations, combinations
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from PIL import Image
from skeletonizer_2d import skeletonizer_2d #custom skeletonizer/graph extractor
import numpy as np
import networkx as nx
from collections import deque
import trimesh
from scipy.optimize import linear_sum_assignment
import random

name = "03-b"
meshname = name + ".stl"
graphname = name  + "_skeleton_graph" + ".graphml"
# xrayname = name + "_xray_original_image.png"
xrayname = name + "_xray_target_image.png"
referncexrayname = name + "_xray_original_image.png"
projected_mask_name = name + "_coarse_reg.png"

# estimagename = name + "_estimated_tform"

current_dir = os.getcwd()
graph_file = os.path.join(current_dir, "graphs", graphname)
mesh_file = os.path.join(current_dir, "meshes", meshname)
projected_mask_file = os.path.join(current_dir, "output_data", projected_mask_name)

xray_file = os.path.join(current_dir, "output_data", xrayname)
xray_file = os.path.abspath(xray_file)
ref_xray_file = os.path.join(current_dir, "output_data", referncexrayname)
print("read xray file:", xray_file)


#=====extracted from print
#directly extracted from generation script
source = np.array([-400, 0, 0])  
detector_point = np.array([100, 0, 0])  #converted to mm
detector_normal = np.array([1, 0, 0]) 
detector_up = np.array([0, 0, -1])
detector_right = np.array([0, -1, 0])
#========

img_width_pixels = 640
img_height_pixels = 320
pixel_pitch = 0.5  # mm / pixel
detector_width_mm = img_width_pixels * pixel_pitch   # 320 mm
detector_height_mm = img_height_pixels * pixel_pitch   # 160 mm

# For the overlay, we choose the detector coordinate system origin at the center of the detector
# That is, (0, 0) in mm corresponds to the center of the detector.
detector_origin_mm = np.array([detector_width_mm / 2, detector_height_mm / 2])


def perspective_projection(points, source, detector_point, detector_normal, detector_up, detector_right):
	""""
	Project 3D points onto a 2D plane defined by the detector normal and up vector.
	
	Args:
		points (numpy.ndarray): 3D points to project (shape: Nx3).
		source (numpy.ndarray): Source point (shape: 1x3).
		detector_point (numpy.ndarray): Point on the detector plane (shape: 1x3).
		detector_normal (numpy.ndarray): Normal vector of the detector plane (shape: 1x3).
		detector_up (numpy.ndarray): Up vector of the detector plane (shape: 1x3).
		detector_right (numpy.ndarray): Right vector of the detector plane (shape: 1x3).

	"""


	# Ray Paramaterization
	# S.T. lambda is the scaling factor for the ray from the source to the points
	# R(lambda) = source + lambda * (points - source)

	# solving for lambda using the intersection with the detector plane
	# lambda = (detector_normal DOT (detector_point - source)) / (detector_normal DOT (points - source))
	lam = np.dot((detector_point - source), detector_normal) / np.dot((points - source), detector_normal)

	# Reshape lam to ensure proper broadcasting
	lam = lam[:, np.newaxis]

	# actual projection
	# projected_points = source + lambda * (points - source)
	projected_points = source + lam * (points - source)

	# mapping to the detector/image coordinates
	# image_x = detector_right DOT (projected_points - detector_point)
	# image_y = detector_up DOT (projected_points - detector_point)
	image_x = np.dot((projected_points - detector_point), detector_right)
	image_y = np.dot((projected_points - detector_point), detector_up)
	return np.column_stack((image_x, image_y))


# 2D graph extraction
skel = skeletonizer_2d(xray_file)
graph_2d = skel.get_graph()

def get_terminal_points(points_2d):
	"""
	Extract terminal points (nodes with degree 1) from a 2D graph.

	Args:
		points_2d (networkx.Graph): A 2D graph.

	Returns:
		np.ndarray: Array of terminal points as (x, y) coordinates.
	"""
	terminal_points = []
	for node, degree in points_2d.degree():
		if degree == 1:  # Terminal points have degree 1
			terminal_points.append(points_2d.nodes[node]['coord'])
	return np.array(terminal_points)

terminal_points_2d = get_terminal_points(graph_2d)
print("Terminal points in 2D:", terminal_points_2d)


def collapse_graph(G):
	"""
	Removes all nodes of degree 2 so that only terminal and branch nodes remain
	"""
	queue = deque(n for n, d in G.degree() if d == 2)

	print("Removing nodes of degree 2...")
	# Add a status indicator for progress
	while queue:
		x = queue.popleft()
	
		# Skip if already removed or no longer deg 2      
		if x in G and G.degree(x) == 2:

			u, v = list(G.neighbors(x)) 

			# Connect u and v directly if not already connected
			if not G.has_edge(u, v):
				G.add_edge(u, v)

			G.remove_node(x)

			# If u or v become degree-2, add them for removal
			for nbr in (u, v):
				if G.degree(nbr) == 2:
					queue.append(nbr)

	mapping = {old: new for new, old in enumerate(G.nodes())}
	G = nx.relabel_nodes(G, mapping)

	print("Complete!")

	return G

def sample_graph(graph, num_samples):
		"""
		Sample points from an nx graph and return as a numpy array.

		Args:
			graph (networkx.Graph): The input graph.
			num_samples (int): Number of points to sample.

		Returns:
			numpy.ndarray: Array of sampled points (shape: num_samples x 3).
		"""
		nodes = list(graph.nodes(data=True))
		sampled_points = []

		for _ in range(num_samples):
			node, data = random.choice(nodes)
			if 'x' in data and 'y' in data and 'z' in data:
				sampled_points.append([data['x'], data['y'], data['z']])

		return np.array(sampled_points)
# 3D graph extraction

def generate_mask_from_projection(points_2d, image_size, radius=1):
	"""
	Generates a binary mask from 2D projected points where each point
	contributes a region of influence (radius).

	Args:
		points_2d (np.ndarray): Nx2 array of (x, y) projected points.
		image_size (tuple): (height, width) of the output mask.
		radius (int): The radius of influence for each projected point.

	Returns:
		mask (np.ndarray): Binary mask of shape (height, width).
	"""
	height, width = image_size
	mask = np.zeros((height, width), dtype=np.uint8)

	# Iterate over the points and set a region of influence
	for pt in points_2d:
		x, y = int(pt[0]), int(pt[1])
		
		# Make sure coordinates are within the bounds of the image
		for i in range(max(0, x - radius), min(width, x + radius + 1)):
			for j in range(max(0, y - radius), min(height, y + radius + 1)):
				# Using Euclidean distance to check influence range
				if (i - x)**2 + (j - y)**2 <= radius**2:
					mask[j, i] = 1  # Set the pixel in the mask to 1 (foreground)

	return mask

# loading the graph points from fileq
graph_3d = nx.read_graphml(graph_file)

def get_terminal_points_3d(graph):
	"""
	Identify terminal points (nodes with degree 1) from a 3D graph.

	Args:
		graph (networkx.Graph): A 3D graph.

	Returns:
		list: List of terminal points as (x, y, z) coordinates.
	"""
	terminal_points = []
	for node, degree in graph.degree():
		if degree == 1:  # Terminal points have degree 1
			terminal_points.append([graph.nodes[node]['x'], graph.nodes[node]['y'], graph.nodes[node]['z']])
	return np.array(terminal_points)
terminal_points_3d = get_terminal_points_3d(graph_3d)
print("Terminal points in 3D:", terminal_points_3d)

def estimate_intrinsics_from_geometry(emitter_position, detector_point, pixel_pitch, image_resolution):
	"""
	Estimate the intrinsic camera matrix using emitter position, detector point, and pixel pitch.
	
	Args:
		emitter_position (np.ndarray): The position of the emitter (camera) (shape: 1x3).
		detector_point (np.ndarray): The point on the detector (shape: 1x3).
		pixel_pitch (float): Pixel pitch in real-world units (e.g., mm/pixel).
		image_resolution (tuple): Image resolution as (width, height) in pixels.
		
	Returns:
		K (np.ndarray): The intrinsic camera matrix.
	"""
	# Step 1: Calculate the focal length (distance between emitter and detector)
	focal_length = np.linalg.norm(emitter_position - detector_point)
	
	# Step 2: Calculate focal lengths in pixel units
	f_x = f_y = focal_length / pixel_pitch  # Assuming square pixels
	
	# Step 3: Calculate principal point (assuming detector center as principal point)
	c_x = image_resolution[0] / 2
	c_y = image_resolution[1] / 2
	
	# Step 4: Build the intrinsic matrix K
	K = np.array([
		[f_x, 0, c_x],
		[0, f_y, c_y],
		[0, 0, 1]
	])
	
	return K

K = estimate_intrinsics_from_geometry(source, detector_point, pixel_pitch, (img_width_pixels, img_height_pixels))
print("Intrinsic matrix K:")
print(K)

mesh = trimesh.load(mesh_file, process=False)  # Load the mesh without processing
mesh.apply_translation(-mesh.bounding_box.centroid)


mesh_points = np.array(mesh.vertices)

def compute_emd(points1, points2):
	"""
	Compute Earth Mover’s Distance (EMD) between two 2D point clouds.
	points1: np.array of shape (N, 2)
	points2: np.array of shape (M, 2)
	"""

	N = points1.shape[0]
	M = points2.shape[0]

	# Uniform weights (assumes all points are equally important)
	a = np.ones((N,)) / N
	b = np.ones((M,)) / M

	# Cost matrix: pairwise Euclidean distances
	C = ot.dist(points1, points2, metric='euclidean')

	# Solve the EMD optimization problem
	emd_value = ot.emd2(a, b, C)  # This returns the optimal EMD cost (float)

	return emd_value

def sliced_wasserstein_distance(x, y, num_projections=100):
	"""
	Computes the sliced Wasserstein distance between two 2D point clouds.

	Args:
		x: Tensor of shape [N, 2]
		y: Tensor of shape [M, 2]
		num_projections: Number of random directions to project onto

	Returns:
		swd: Scalar tensor representing the sliced Wasserstein distance
	"""
	assert x.shape[1] == y.shape[1] == 2, "Only supports 2D point sets."


	# Normalize both sets (optional but recommended for consistency)
	x = x - x.mean(dim=0)
	y = y - y.mean(dim=0)

	# Sample random directions from the unit circle
	theta = torch.rand(num_projections) * 2 * np.pi
	directions = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # [num_projections, 2]

	# Project both point sets onto each direction
	proj_x = x @ directions.T  # [N, num_projections]
	proj_y = y @ directions.T  # [M, num_projections]

	# Sort projections along each direction
	proj_x_sorted, _ = torch.sort(proj_x, dim=0)
	proj_y_sorted, _ = torch.sort(proj_y, dim=0)

	# Match min length for fair comparison
	min_len = min(proj_x_sorted.shape[0], proj_y_sorted.shape[0])
	proj_x_sorted = proj_x_sorted[:min_len]
	proj_y_sorted = proj_y_sorted[:min_len]

	# Compute 1D Wasserstein distance per direction
	distances = torch.abs(proj_x_sorted - proj_y_sorted).mean(dim=0)  # [num_projections]

	# Return average over projections
	return distances.mean()

def mse_hungarian(X, Y):
	dists = np.linalg.norm(X[:, None] - Y[None, :], axis=-1)  # [N, N]
	row_ind, col_ind = linear_sum_assignment(dists)
	matched_X = X[row_ind]
	matched_Y = Y[col_ind]
	return np.mean(np.sum((matched_X - matched_Y) ** 2, axis=1))

def cpd_error_metric(X_target_2d, Y_projected_2d):
	"""
	Computes the CPD alignment error between two point sets.

	Args:
		X_target_2d (np.ndarray): The 2D ground truth point cloud. Shape: [N, 2]
		Y_projected_2d (np.ndarray): The 2D projected point cloud. Shape: [N, 2]

	Returns:
		float: The MSE between aligned points (TY) and target after CPD.
	"""
	reg = RigidRegistration(X=X_target_2d, Y=Y_projected_2d)
	TY, _ = reg.register()

	# Compute Mean Squared Error as the final metric
	mse = np.mean(np.linalg.norm(X_target_2d - TY, axis=1) ** 2)
	return mse

def combined_hungarian_angle_metric(X1, X2, angle_weight=0.1):
	"""
	Computes combined loss: Hungarian MSE + weighted angle deviation.
	"""

	def pca_angle(vecs):
		"""Compute the angle of the first principal component in degrees [0, 180)."""
		cov = np.cov(vecs.T)
		eigvals, eigvecs = np.linalg.eigh(cov)
		principal_axis = eigvecs[:, np.argmax(eigvals)]
		angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
		return np.degrees(angle_rad) % 180

	def angle_penalty_pca(X1, X2):
		"""Compute minimal absolute angular deviation between PCA principal axes."""
		angle1 = pca_angle(X1)
		angle2 = pca_angle(X2)
		diff = np.abs(angle1 - angle2)
		return min(diff, 180 - diff)

	def hungarian_mse(X1, X2):
		"""Compute MSE between matched points using the Hungarian algorithm."""
		dists = np.linalg.norm(X1[:, None, :] - X2[None, :, :], axis=2)
		row_ind, col_ind = linear_sum_assignment(dists)
		matched_X1 = X1[row_ind]
		matched_X2 = X2[col_ind]
		mse = np.mean(np.sum((matched_X1 - matched_X2) ** 2, axis=1))
		return mse
	mse = hungarian_mse(X1, X2)
	angle_error = angle_penalty_pca(X1, X2)
	total_loss = mse + angle_weight * angle_error
	return total_loss

def dice_coefficient(mask1, mask2):
	intersection = np.logical_and(mask1, mask2).sum()
	return 2. * intersection / (mask1.sum() + mask2.sum() + 1e-8)

def generate_so3_grid(n_axis=30, n_spin=int(360/10)):
	"""
	Returns a list of rotation matrices sampling SO(3) by:
	 - n_axis directions on S^2 (Fibonacci)
	 - n_spin equally‐spaced spins around each axis.
	Total samples = n_axis * n_spin.
	"""

	def fibonacci_sphere(n_points):
		"""
		Return n_points approximately uniformly distributed on the unit 2‑sphere.
		"""
		points = []
		phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
		for i in range(n_points):
			y = 1 - (i / float(n_points - 1)) * 2      # y goes from  1 to -1
			radius = np.sqrt(1 - y*y)                 # radius at that y
			theta = phi * i                           # golden angle increment

			x = np.cos(theta) * radius
			z = np.sin(theta) * radius
			points.append([x, y, z])
		return np.array(points)

	axes = fibonacci_sphere(n_axis)
	R_matrices = []

	for u in axes:
		for k in range(n_spin):
			theta = 2 * np.pi * k / n_spin
			# axis‐angle to quaternion [x,y,z,w]
			q_xyz = u * np.sin(theta/2)
			q_w   = np.cos(theta/2)
			quat  = np.hstack((q_xyz, q_w))
			Rm    = R.from_quat(quat).as_matrix()
			R_matrices.append(Rm)

	return R_matrices

def iterative_rotation_projection(vertices_2d, vertices_3d):
	
	rotation_matrices = generate_so3_grid()

	best_error = float('inf')
	best_rotation = None
	chamfer = ChamferDistance()
	best_TY = None
	centroid = np.array([0, 0, 0])

	# testing identity rotation
	identity = np.eye(3)
	transformed_3d = np.dot(vertices_3d, identity.T)
	rotated_centroid = np.dot(centroid, identity.T)

	transformed_3d = transformed_3d - rotated_centroid

	projected_points_3d = perspective_projection(transformed_3d, source, detector_point, detector_normal, detector_up, detector_right)

	# for visual alignment
	projected_points_3d = projected_points_3d + detector_origin_mm
	projected_points_3d = projected_points_3d / pixel_pitch
	projected_points_3d[:,1] = img_height_pixels - projected_points_3d[:,1]
	
	
	dist = combined_hungarian_angle_metric(vertices_2d, projected_points_3d)

	print("!!!Error for identity rotation:", dist)


	for rot in tqdm(rotation_matrices):

		# Apply rotation to the 3D vertices
		transformed_3d = np.dot(vertices_3d, rot.T)
		rotated_centroid = np.dot(centroid, rot.T)

		transformed_3d = transformed_3d - rotated_centroid

		projected_points_3d = perspective_projection(transformed_3d, source, detector_point, detector_normal, detector_up, detector_right)
	
		# for visual alignment
		projected_points_3d = projected_points_3d + detector_origin_mm
		projected_points_3d = projected_points_3d / pixel_pitch
		projected_points_3d[:,1] = img_height_pixels - projected_points_3d[:,1]

		# Load the X-ray image
		# xray_image = Image.open(xray_file)

		# -------------------------------
		# # Display the overlay
		# plt.figure(figsize=(10, 8))

		# # Set the extent of the image in physical mm converted to pixels.
		# # Since the image is 640x320, we set extent = [0, 640, 320, 0] to have (0,0) at top-left.
		# plt.imshow(xray_image, cmap="gray", extent=[0, img_width_pixels, img_height_pixels, 0])
		# plt.scatter(projected_points_3d[:, 0], projected_points_3d[:, 1], c='b', s=1, label="Mesh Points")

		# # Plot edges
		# for (i, j) in graph_2d.edges():
		# 	xi, yi = graph_2d.nodes[i]['coord']
		# 	xj, yj = graph_2d.nodes[j]['coord']
		# 	plt.plot([yi, yj], [xi, xj], 'g-', linewidth=2, label="2D Graph Edges")

		# # Plot nodes
		# for i, data in graph_2d.nodes(data=True):
		# 	x, y = data['coord']
		# 	plt.plot(y, x, 'o', markersize=5, c="y", label="2D Graph Nodes")


		# plt.xlabel('Pixel X')
		# plt.ylabel('Pixel Y')
		# plt.title('Projected 3D Points Superimposed on X-ray Image')
		# plt.legend()
		# plt.grid(True)
		# plt.show()



		# Initialize the registration
		# reg = RigidRegistration(X=vertices_2d, Y=projected_points_3d)

		# # Run the registration
		# TY, (s_reg, R_reg, t_reg) = reg.register()
		# Compute Chamfer Distance using PyTorch tensors
		
		

		dist = combined_hungarian_angle_metric(vertices_2d, projected_points_3d)

		

		if dist < best_error:
			best_error = dist
			best_rotation = rot
			best_TY = projected_points_3d
			print("New best error found:", best_error)
			print("Best rotation matrix:", best_rotation)

	
	return best_rotation, best_error, best_TY

def iterative_rotation_projection_visual( mesh_points, img):
	
	rotation_matrices = generate_so3_grid()

	img = np.array(img)
	original_mask = np.where((img > 0) & (img <= 254), 1, 0).astype(np.uint8)


	best_error = 0.
	best_rotation = None
	best_mask = None
	centroid = np.array([0, 0, 0])

	# testing identity rotation
	identity = np.eye(3)
	transformed_3d = np.dot(mesh_points, identity.T)
	rotated_centroid = np.dot(centroid, identity.T)

	transformed_3d = transformed_3d - rotated_centroid

	projected_points_3d = perspective_projection(transformed_3d, source, detector_point, detector_normal, detector_up, detector_right)

	# for visual alignment
	projected_points_3d = projected_points_3d + detector_origin_mm
	projected_points_3d = projected_points_3d / pixel_pitch
	projected_points_3d[:,1] = img_height_pixels - projected_points_3d[:,1]
	

	mesh_mask = generate_mask_from_projection(projected_points_3d, (img_height_pixels, img_width_pixels))
	
	dist = dice_coefficient(original_mask, mesh_mask)

	print("!!!Error for identity rotation:", dist)


	for rot in tqdm(rotation_matrices):

		# Apply rotation to the 3D vertices
		transformed_3d = np.dot(mesh_points, rot.T)
		rotated_centroid = np.dot(centroid, rot.T)

		transformed_3d = transformed_3d - rotated_centroid

		projected_points_3d = perspective_projection(transformed_3d, source, detector_point, detector_normal, detector_up, detector_right)
	
		# for visual alignment
		projected_points_3d = projected_points_3d + detector_origin_mm
		projected_points_3d = projected_points_3d / pixel_pitch
		projected_points_3d[:,1] = img_height_pixels - projected_points_3d[:,1]
		mesh_mask = generate_mask_from_projection(projected_points_3d, (img_height_pixels, img_width_pixels))

		# -------------------------------
		# plt.imshow(img, cmap='gray')
		# plt.imshow(mesh_mask, cmap='gray', alpha = 0.5)
		
		# plt.imshow(original_mask, cmap='gray', alpha = 0.5)
		# plt.show()


		# Initialize the registration
		# reg = RigidRegistration(X=vertices_2d, Y=projected_points_3d)

		# # Run the registration
		# TY, (s_reg, R_reg, t_reg) = reg.register()
		# Compute Chamfer Distance using PyTorch tensors
		
		

		dist = dice_coefficient(original_mask, mesh_mask)

		

		if dist > best_error:
			best_error = dist
			best_rotation = rot
			best_mask = mesh_mask
			print("New best error found:", best_error)
			print("Best rotation matrix:", best_rotation)

			# plt.imshow(img, cmap='gray')
			# plt.imshow(mesh_mask, cmap='gray', alpha = 0.5)
			
			# plt.imshow(original_mask, cmap='gray', alpha = 0.5)
			# plt.show()


	
	return best_rotation, best_error, best_mask

def find_best_pose_from_unordered_points(P3, P2, K, xray_image, mesh_points):
	best_dice_score = 0.
	best_rvec = None
	best_tvec = None
	best_perm = None
	best_mask = None
	centroid = np.array([0, 0, 0])


	img = np.array(xray_image)
	original_mask = np.where((img > 0) & (img <= 254), 1, 0).astype(np.uint8)
	
	n3, n2 = len(P3), len(P2)
	if n3 < 4 or n2 < 4:
		raise ValueError("Both sets must have at least 4 points for PnP.")

	best_inliers = -1
	best_result = None

	if n3 <= n2:
		small_set_3d, large_set_2d = P3, P2
		is_3d_smaller = True
	else:
		small_set_3d, large_set_2d = P2, P3
		is_3d_smaller = False

	k = len(small_set_3d)

	  # Try all combinations of size k from the larger set
	for idxs in tqdm(combinations(range(len(large_set_2d)), k)):
		subset = large_set_2d[list(idxs)]

		# Test all permutations
		for perm in permutations(range(k)):
			permuted = subset[list(perm)]

			if is_3d_smaller:
				P3_test = small_set_3d.reshape(-1, 3)
				P2_test = permuted.reshape(-1, 2)
			else:
				P3_test = permuted.reshape(-1, 3)
				P2_test = small_set_3d.reshape(-1, 2)

			P3 = P3_test.reshape(-1, 1, 3).astype(np.float32)
			P2 = P2_test.reshape(-1, 1, 2).astype(np.float32)

			success, rvec, tvec, inliers = cv2.solvePnPRansac(
				objectPoints = P3.astype(np.float32),
				imagePoints  = P2.astype(np.float32),
				cameraMatrix = K, distCoeffs=None,
				reprojectionError=10.0,  # Allow some noise in pixels
				flags=cv2.SOLVEPNP_EPNP
			)


			if success:

				#apply rotation to mesh points
				rotation_matrix, _ = cv2.Rodrigues(rvec)
				transformed_3d = np.dot(mesh_points, rotation_matrix.T)
				rotated_centroid = np.dot(centroid, rotation_matrix.T)

				transformed_3d = transformed_3d - rotated_centroid

				projected_points_3d = perspective_projection(transformed_3d, source, detector_point, detector_normal, detector_up, detector_right)
			
				# for visual alignment
				projected_points_3d = projected_points_3d + detector_origin_mm
				projected_points_3d = projected_points_3d / pixel_pitch
				projected_points_3d[:,1] = img_height_pixels - projected_points_3d[:,1]
				mesh_mask = generate_mask_from_projection(projected_points_3d, (img_height_pixels, img_width_pixels))
				
			
				dice_score = dice_coefficient(original_mask, mesh_mask)

				if dice_score > best_dice_score:
					print("New best dice score found:", dice_score)
					best_dice_score = dice_score
					best_rvec = rvec
					best_tvec = tvec
					best_perm = perm
					best_mask = mesh_mask
		
	if best_dice_score == 0:
		print("No valid pose found.")
		return None, None, None, None, None
	return best_rvec, best_tvec, best_perm, best_dice_score, best_mask



def find_closest_indices(projected_points, mask):
	"""
	Find the indices of the points in `projected_points` that are closest to the non-zero pixels in `mask`.

	Args:
		projected_points (np.ndarray): Nx2 array of 2D points.
		mask (np.ndarray): Binary mask of shape (height, width).

	Returns:
		list: Indices of the closest points in `projected_points` to the non-zero pixels in `mask`.
	"""
	non_zero_pixels = np.column_stack(np.nonzero(mask))  # Get (y, x) coordinates of non-zero pixels
	closest_indices = []

	for point in projected_points:
		distances = np.linalg.norm(non_zero_pixels - point[::-1], axis=1)  # Compute distances
		closest_indices.append(np.argmin(distances))  # Find the index of the closest pixel

	return closest_indices

# apply an arbitrary rotation and project
def generate_random_rotation_matrix():
	"""
	Generate a random 3x3 rotation matrix.
	"""
	random_quaternion = np.random.randn(4)
	random_quaternion /= np.linalg.norm(random_quaternion)  # Normalize to unit quaternion
	rotation_matrix = R.from_quat(random_quaternion).as_matrix()
	return rotation_matrix

transformed_3d = np.dot(mesh_points, np.eye(3).T)

# transformed_3d = transformed_3d - rotated_centroid
projected_mesh_points = perspective_projection(transformed_3d, source, detector_point, detector_normal, detector_up, detector_right)

# for visual alignment
projected_mesh_points = projected_mesh_points + detector_origin_mm
projected_mesh_points = projected_mesh_points / pixel_pitch
projected_mesh_points[:,1] = img_height_pixels - projected_mesh_points[:,1]
mesh_mask = generate_mask_from_projection(projected_mesh_points, (img_height_pixels, img_width_pixels))


# Save the mesh_mask to the coarse_mask_file
plt.imsave(projected_mask_file, mesh_mask, cmap='gray')
print(f"projected mesh mask saved to: {projected_mask_file}")

print("skeletonizing projected mask...")
projected_skeleton = skeletonizer_2d(projected_mask_file, invert=True)
projected_graph_2d = projected_skeleton.get_graph()
projected_term_points_2d = get_terminal_points(projected_graph_2d)
num_term_points = projected_term_points_2d.shape[0]

# Find the indices of the closest points
closest_indices = find_closest_indices(projected_term_points_2d, mesh_mask)

terminal_points_3d = mesh_points[closest_indices]

xray_image = Image.open(xray_file)

best_rvec, best_tvec, best_perm, best_dice_score, best_mask = find_best_pose_from_unordered_points(terminal_points_3d, terminal_points_2d, K, xray_image, mesh_points)

rotation_matrix, _ = cv2.Rodrigues(best_rvec)
homogeneous_matrix = np.eye(4)
homogeneous_matrix[:3, :3] = rotation_matrix

print("best rotation matrix", homogeneous_matrix)
print("best dice score", best_dice_score)


plt.imshow(xray_image, cmap='gray')
plt.imshow(best_mask, cmap='gray', alpha= 0.5)

output_folder = os.path.join(current_dir, "output_data")
os.makedirs(output_folder, exist_ok=True)


# # Plot the skeletonized map with the graph connections
plt.imshow(xray_image, cmap='gray')

plt.imshow(best_mask, cmap='gray', alpha = 0.5)
plt.axis('off')
plt.show()

xgen = XRayDepthGenerator(name, "endpoint")
xgen.apply_rotation(homogeneous_matrix)
xgen.get_xray_and_depth()


print("rotation matrix:", homogeneous_matrix)
print("dice score:", best_dice_score)