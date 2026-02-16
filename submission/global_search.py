from xray_depth_generator import XRayDepthGenerator
import numpy as np
from tqdm import tqdm
import random
from scipy.spatial.transform import Rotation as R
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from PIL import Image
import numpy as np
import networkx as nx
from collections import deque
import trimesh

name = "03-b"
meshname = name + ".stl"
graphname = name  + "_skeleton_graph" + ".graphml"
# xrayname = name + "_xray_original_image.png"
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
	# focal length (distance between emitter and detector)
	focal_length = np.linalg.norm(emitter_position - detector_point)
	
	# convert focal length to pixels
	f_x = f_y = focal_length / pixel_pitch  # Assuming square pixels
	
	# Calculate principal point (assuming detector center as principal point)
	c_x = image_resolution[0] / 2
	c_y = image_resolution[1] / 2
	
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




def dice_coefficient(mask1, mask2):
	intersection = np.logical_and(mask1, mask2).sum()
	return 2. * intersection / (mask1.sum() + mask2.sum() + 1e-8)


def generate_rotation_matrices(angle_step=30, degrees=True):
	"""
	Generate rotation matrices by systematically iterating through angles
	around the x, y, and z axes.

	Parameters:
	- angle_step (float): Step size for angles in degrees or radians.
	- degrees (bool): If True, angles are in degrees; if False, in radians.

	Returns:
	- List of 3x3 rotation matrices as NumPy arrays.
	"""
	# Define angle range based on the specified step
	if degrees:
		angles = np.arange(0, 360, angle_step)
	else:
		angles = np.arange(0, 2 * np.pi, angle_step)

	rotation_matrices = []

	# Iterate through all combinations of angles for x, y, z axes
	for angles_xyz in product(angles, repeat=3):
		# Create a rotation object using Euler angles
		rotation = R.from_euler('xyz', angles_xyz, degrees=degrees)
		# Convert to rotation matrix and append to the list
		rotation_matrices.append(rotation.as_matrix())

	return rotation_matrices



def iterative_rotation_projection_visual( mesh_points, img):
	"""
	Iteratively apply rotations to the mesh points and project them onto 
	the image plane and then compare with the original image using a 
	dice coefficient.

	Parameters:
	- mesh_points (numpy.ndarray): 3D points of the mesh (shape: Nx3).
	- img (numpy.ndarray): 2D image to compare against (shape: HxW).

	Returns:
	- best_rotation (numpy.ndarray): The rotation matrix that gives the best match.
	- best_error (float): The best error (dice coefficient).
	- best_mask (numpy.ndarray): The mask generated from the best rotation.
	"""
	
	rotation_matrices = generate_rotation_matrices()

	img = np.array(img)
	original_mask = np.where((img > 0) & (img <= 254), 1, 0).astype(np.uint8)


	best_dice = 0.
	best_rotation = None
	best_mask = None
	centroid = np.array([0, 0, 0])

	# testing identity rotation
	identity = np.eye(3)
	transformed_3d = np.dot(mesh_points, identity.T)
	rotated_centroid = np.dot(centroid, identity.T)

	transformed_3d = transformed_3d - rotated_centroid

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

		dice = dice_coefficient(original_mask, mesh_mask)

		if dice > best_dice:
			best_dice = dice
			best_rotation = rot
			best_mask = mesh_mask
			print("New best error found:", best_dice)
			print("Best rotation matrix:", best_rotation)

	return best_rotation, best_dice, best_mask


xray_image = Image.open(xray_file)

mesh = trimesh.load(mesh_file, process=False)  # Load the mesh without processing
mesh.apply_translation(-mesh.bounding_box.centroid)
mesh_points = np.array(mesh.vertices)

best_rotation, best_dice, best_mask = iterative_rotation_projection_visual(mesh_points, xray_image)

# Best rotation matrix: [[ 0.8660254 -0.5        0.       ]
#  [ 0.5        0.8660254  0.       ]
#  [ 0.         0.         1.       ]]
# Best error: 0.44510693454280936
homogeneous_matrix = np.eye(4)
homogeneous_matrix[:3, :3] = best_rotation



print("Best rotation matrix:", homogeneous_matrix)
print("Best dice:", best_dice)

output_folder = os.path.join(current_dir, "output_data")
os.makedirs(output_folder, exist_ok=True)

plt.imsave(coarse_mask_file, best_mask, cmap='gray')

print(f"Best global search mask was saved to: {coarse_mask_file}")

print("showing estimated rotation overlaid on original xray...")

plt.imshow(xray_image, cmap='gray')

plt.imshow(best_mask, cmap='gray', alpha = 0.5)

plt.axis('off')
plt.show()




xgen = XRayDepthGenerator(name, "global")
xgen.apply_rotation(homogeneous_matrix)
xgen.get_xray_and_depth()


print("rotation matrix:", homogeneous_matrix)
print("dice score:", best_dice)