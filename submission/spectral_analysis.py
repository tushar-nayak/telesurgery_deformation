from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot  # POT library
import numpy as np
from tqdm import tqdm
import random
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from PIL import Image
from skeletonizer_2d import skeletonizer_2d #custom skeletonizer/graph extractor
from xray_depth_generator import XRayDepthGenerator
import numpy as np
import networkx as nx
from collections import deque
import trimesh
from sklearn.manifold import SpectralEmbedding

name = "03-b"
meshname = name + ".stl"
graphname = name  + "_skeleton_graph" + ".graphml"
xrayname = name + "_xray_target_image.png"
referncexrayname = name + "_xray_original_image.png"
coarse_mask_name = name + "_coarse_reg.png"


current_dir = os.getcwd()
graph_file = os.path.join(current_dir, "graphs", graphname)
mesh_file = os.path.join(current_dir, "meshes", meshname)
coarse_mask_file = os.path.join(current_dir, "output_data", coarse_mask_name)

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

# detector is at the center of the image plane
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


def gw_correspondences_no_graph(pts2d, pts3d_proj, reg_epsilon=.5):
	"""
	Compute 2D–3D correspondences via Gromov–Wasserstein on Euclidean distances.

	Args:
	  pts2d        (np.ndarray): [n2, 2] skeleton keypoints in the X‑ray image.
	  pts3d_proj   (np.ndarray): [n3, 2] projected skeleton keypoints from the 3D mesh.
	  reg_epsilon  (float): entropic regularization weight.

	Returns:
	  matches2d    (np.ndarray): indices 0..n2-1 of 2D points.
	  matches3d    (np.ndarray): for each 2D index, the best 3D index.
	  T            (np.ndarray): full GW coupling matrix of shape (n3, n2).
	"""
	# 1) Build pairwise distance matrices (C1 for 3D, C2 for 2D)
	C3 = cdist(pts3d_proj, pts3d_proj, metric='euclidean')
	C2 = cdist(pts2d,      pts2d,       metric='euclidean')

	# 2) Uniform histograms
	n3, n2 = C3.shape[0], C2.shape[0]
	p = np.ones((n3,)) / n3
	q = np.ones((n2,)) / n2

	# 3) Solve entropic GW (more stable than non‑entropic)
	T = ot.gromov.entropic_gromov_wasserstein(
		C1=C3, C2=C2, p=p, q=q, loss_fun='square_loss', epsilon=reg_epsilon,
		max_iter=1000
	)

	# 4) Hard assignment: for each 2D point j, pick the 3D i with max coupling
	matches2d = np.arange(n2)
	matches3d = np.argmax(T, axis=0)

	return matches2d, matches3d, T



#============ spectral analysis====================

K = estimate_intrinsics_from_geometry(source, detector_point, pixel_pitch, (img_width_pixels, img_height_pixels))
# print("Intrinsic matrix K:")
# print(K)

# 2D graph extraction
skel = skeletonizer_2d(xray_file)
graph_2d = skel.get_graph()
skel.plot_graph()

graph_3d = nx.read_graphml(graph_file)
graph_3d = collapse_graph(graph_3d)

# Convert graphs to adjacency matrices
adj_2d = nx.to_numpy_array(graph_2d)
adj_3d = nx.to_numpy_array(graph_3d)
# print("Adjacency matrix 2D :", adj_2d)

# Compute Spectral Embeddings
embedding = SpectralEmbedding(n_components=5, affinity='precomputed')
embedding_2d = embedding.fit_transform(adj_2d)
embedding_3d = embedding.fit_transform(adj_3d)

# print("2D embedding :", embedding_2d)
# print("3D embedding :", embedding_3d)

# -- Hungarian problem -- 
# Compute pairwise distances between embeddings
cost_matrix = cdist(embedding_2d, embedding_3d, metric='euclidean')

# Solve the assignment problem
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Create a mapping from 2d to 3d nodes
node_mapping = dict(zip(row_ind, col_ind))
print(node_mapping)

ordered_2d = None
ordered_3d = None

for ind_2d, ind_3d in node_mapping.items():
	# Retrieve coordinates from node attributes
	coord_2d = np.array([graph_2d.nodes[ind_2d]['coord'][0], graph_2d.nodes[ind_2d]['coord'][1]]).astype(np.float32)
	coord_3d = np.array([graph_3d.nodes[ind_3d]['x'], graph_3d.nodes[ind_3d]['y'], graph_3d.nodes[ind_3d]['z'] ]).astype(np.float32)

	if ordered_2d is None:
		ordered_2d = coord_2d
		ordered_3d = coord_3d
	else:
		ordered_2d = np.vstack((ordered_2d, coord_2d))
		ordered_3d = np.vstack((ordered_3d, coord_3d))
#============ spectral analysis====================

success, rvec, tvec, inliers = cv2.solvePnPRansac(
	objectPoints = ordered_3d.reshape(-1,1,3).astype(np.float32),
	imagePoints  = ordered_2d.reshape(-1,1,2).astype(np.float32),
	cameraMatrix = K, distCoeffs=None,
	reprojectionError=10.0,  # Allow some noise in pixels
	flags=cv2.SOLVEPNP_P3P
)


if not success:
	raise ValueError("PnP did not succeed. ")

# Convert rvec to a rotation matrix
rotation_matrix, _ = cv2.Rodrigues(rvec)
homogeneous_matrix = np.eye(4)
homogeneous_matrix[:3, :3] = rotation_matrix



#for dice score calculation
centroid = np.array([0, 0, 0])
mesh = trimesh.load(mesh_file, process=False)  # Load the mesh without processing
mesh.apply_translation(-mesh.bounding_box.centroid)
mesh_points = np.array(mesh.vertices)

transformed_3d = np.dot(mesh_points, rotation_matrix.T)
rotated_centroid = np.dot(centroid, rotation_matrix.T)

transformed_3d = transformed_3d - rotated_centroid

projected_points_3d = perspective_projection(transformed_3d, source, detector_point, detector_normal, detector_up, detector_right)

# for visual alignment
projected_points_3d = projected_points_3d + detector_origin_mm
projected_points_3d = projected_points_3d / pixel_pitch
projected_points_3d[:,1] = img_height_pixels - projected_points_3d[:,1]
mesh_mask = generate_mask_from_projection(projected_points_3d, (img_height_pixels, img_width_pixels))

xray_image = Image.open(xray_file)
xray_image = np.array(xray_image)
original_mask = np.where((xray_image > 0) & (xray_image <= 254), 1, 0).astype(np.uint8)
dice_score = dice_coefficient(original_mask, mesh_mask)

plt.imshow(xray_image, cmap='gray')
plt.imshow(mesh_mask, cmap='gray', alpha = 0.5)
plt.axis('off')
print("showing estimated rotation overlaid on original xray...")
plt.show()


xgen = XRayDepthGenerator(name, "spectral")
xgen.apply_rotation(homogeneous_matrix)
xgen.get_xray_and_depth()


print("rotation matrix:", homogeneous_matrix)
print("dice score:", dice_score)