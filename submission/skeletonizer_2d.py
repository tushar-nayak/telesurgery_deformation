# a class definition based on skeleton_v2-2d.py

import itertools
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import networkx as nx
from scipy.ndimage import convolve
from collections import deque



class skeletonizer_2d:
	"""
	Class to perform skeletonization on 2D images.
	"""

	def __init__(self, image_path, invert=False):
		self.image_path = image_path
		if not os.path.exists(self.image_path):
			raise FileNotFoundError(f"The specified image path does not exist: {self.image_path}")
		self.img = sitk.ReadImage(self.image_path, sitk.sitkUInt8)
		if invert:
			self.img = sitk.InvertIntensity(self.img, maximum=255)


		self.skeleton_np = None
		self.graph = None
		self.points = None

	def find_nearest_branch(self, skeleton, start_coord, branching_coords):
		"""
		Explore the skeleton path starting from a given coordinate until
		a branching or terminal point is reached.

		Parameters:
		- skeleton: The skeleton image as a NumPy array.
		- start_coord: The starting coordinate (row, col) as a tuple.
		- branching_coords: Array of branching point coordinates.
		- index_offset: Offset to adjust the index of the nearest branching point. Intended to be length of the branching_coords.

		Returns:
		- index of the nearest branching point in the skeleton.
		"""
		
		visited = set()
		queue = deque([(start_coord, [start_coord])])  # Each element is (current_coord, path_so_far)
		visited.add(tuple(start_coord))

		# Precompute branching point lookup
		branching_lookup = {tuple(coord): idx for idx, coord in enumerate(branching_coords)}

		directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
					(-1, -1), (-1, 1), (1, -1), (1, 1)]

		while queue:
			current, path = queue.popleft()

			if current in branching_lookup and current != tuple(start_coord):
				return branching_lookup[current]  # Return index of the nearest branching coord

			for dx, dy in directions:
				neighbor = (current[0] + dx, current[1] + dy)
				if (0 <= neighbor[0] < skeleton.shape[0] and
					0 <= neighbor[1] < skeleton.shape[1] and
					skeleton[neighbor] == 1 and
					neighbor not in visited):
					visited.add(neighbor)
					queue.append((neighbor, path + [neighbor]))

		raise ValueError(f"WARNING: no branching point found for terminal point {start_coord}")


	def find_nearest_branches(self,skeleton, start_coord, branching_coords):
		"""
		Explore the skeleton path starting from a given coordinate until
		a branching or terminal point is reached.

		Parameters:
		- skeleton: The skeleton image as a NumPy array.
		- start_coord: The starting coordinate (row, col) as a tuple.
		- branching_coords: Array of branching point coordinates.
		- index_offset: Offset to adjust the index of the nearest branching point. Intended to be length of the branching_coords.

		Returns:
		- index of the nearest branching point in the skeleton.
		"""
		visited = set()
		queue = deque([start_coord])
		visited.add(tuple(start_coord))
		connected_indices = []

		# Precompute branching point lookup
		branching_lookup = {tuple(coord): idx for idx, coord in enumerate(branching_coords)}

		# 8-connected neighbors
		directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
					(-1, -1), (-1, 1), (1, -1), (1, 1)]

		while queue:
			current = queue.popleft()

			# Skip self when checking branching
			if current in branching_lookup and current != tuple(start_coord):
				idx = branching_lookup[current]
				if idx not in connected_indices:
					connected_indices.append(idx)

				# Do not expand from branching points
				continue

			# Explore neighbors
			for dx, dy in directions:
				neighbor = (current[0] + dx, current[1] + dy)
				if (0 <= neighbor[0] < skeleton.shape[0] and
					0 <= neighbor[1] < skeleton.shape[1] and
					skeleton[neighbor] == 1 and
					neighbor not in visited):


					# Check if the neighbor is not within 3 nodes of any branching point that is already a connected index
					if all(np.linalg.norm(np.array(neighbor) - np.array(branching_coords[idx])) >= 3 for idx in connected_indices):
						visited.add(neighbor)
						queue.append(neighbor)
						

		return connected_indices

	def find_all_connections(self,skeleton, start_coord, branching_coords, terminal_coords):
		"""
		Explore the skeleton path starting from a given coordinate until
		a branching or terminal point is reached.

		Parameters:
		- skeleton: The skeleton image as a NumPy array.
		- start_coord: The starting coordinate (row, col) as a tuple.
		- branching_coords: Array of branching point coordinates.
		- terminal_coords: Array of terminal point coordinates.

		Returns:
		- list of indices that are connected to the starting coordinate
		"""
		visited = set()
		queue = deque([start_coord])
		visited.add(tuple(start_coord))
		connected_indices = []

		# Precompute branching point lookup
		branching_lookup = {tuple(coord): idx for idx, coord in enumerate(branching_coords)}
		terminal_lookup = {tuple(coord): idx+len(branching_coords) for idx, coord in enumerate(terminal_coords)}

		# 8-connected neighbors
		directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
					(-1, -1), (-1, 1), (1, -1), (1, 1)]

		while queue:
			current = queue.popleft()

			# Skip self when checking branching
			if current in branching_lookup and current != tuple(start_coord):
				idx = branching_lookup[current]
				if idx not in connected_indices:
					connected_indices.append(idx)
				# Do not expand from branching points
				continue

			# Skip self when checking branching
			if current in terminal_lookup and current != tuple(start_coord):
				idx = terminal_lookup[current]
				if idx not in connected_indices:
					connected_indices.append(idx)
				# Do not expand from terminal points
				continue

			# Explore neighbors
			for dx, dy in directions:
				neighbor = (current[0] + dx, current[1] + dy)
				if (0 <= neighbor[0] < skeleton.shape[0] and
					0 <= neighbor[1] < skeleton.shape[1] and
					skeleton[neighbor] == 1 and
					neighbor not in visited):
					visited.add(neighbor)
					queue.append(neighbor)

		return connected_indices
				
	def merge_branching_points(self, graph, tolerance=5):
		"""
		Merge branching points in the graph that are within a specified distance.

		Parameters:
		- graph: The input graph with nodes and edges.
		- tolerance: The distance threshold for merging nodes.

		Returns:
		- Merged graph with branching points merged.
		"""
		gr = graph.copy()

		while True:
			merged = False
			for node1, node2 in itertools.combinations(gr.nodes(), 2):
				coord1 = np.array(gr.nodes[node1]['coord'])
				coord2 = np.array(gr.nodes[node2]['coord'])

				if np.linalg.norm(coord1 - coord2) < tolerance:
					neighbors2 = gr.neighbors(node2)
					for neighbor in neighbors2:
						if neighbor != node1:
							gr.add_edge(node1, neighbor)
					
					gr.remove_node(node2)
					
					merged = True
					break  # Restart after each merge

			if not merged:
				break
		
		mapping = {old: new for new, old in enumerate(gr.nodes())}
		gr = nx.relabel_nodes(gr, mapping)
		return gr
	
	def get_points(self, number_of_points=100):
		if self.points is not None:
			return self.points

		print("========= SKELETONIZING 2D IMAGE =========" )
		# Binarization
		bin_img = sitk.BinaryThreshold(self.img, lowerThreshold=0, upperThreshold=254, insideValue=1, outsideValue=0)

		# Skeletonization
		skeleton = sitk.BinaryThinning(bin_img)

		# Convert to NumPy for analysis
		self.skeleton_np = sitk.GetArrayFromImage(skeleton).astype(np.uint8)


		# Define 2D 8-neighborhood kernel (or 3D if needed)
		lg_branch_kernel = np.array([[1,1,1,1,1],
									[1,0,0,0,1],
									[1,0,0,0,1],
									[1,0,0,0,1],
									[1,1,1,1,1]])

		# Define 2D 8-neighborhood kernel (or 3D if needed)
		sm_branch_kernel = np.array([[0,1,0],
									[1,0,1],
									[0,1,0]])

		# Define 2D 8-neighborhood kernel (or 3D if needed)
		terminal_kernel = np.array([[1,1,1],
									[1,0,1],
									[1,1,1]])

		# Count neighbors using convolution
		branch_neighbor_count_1 = convolve(self.skeleton_np, sm_branch_kernel, mode='constant', cval=0)
		branch_neighbor_count_2 = convolve(self.skeleton_np, lg_branch_kernel, mode='constant', cval=0)

		# Terminal points: 1 neighbor
		branching_points_1 = (self.skeleton_np == 1) & (branch_neighbor_count_1 >= 3) 
		branching_points_2 = (self.skeleton_np == 1) & (branch_neighbor_count_2 >= 5)

		# Count neighbors using convolution
		terminal_neighbor_count = convolve(self.skeleton_np, terminal_kernel, mode='constant', cval=0)

		# Terminal points: 1 neighbor
		terminal_points = (self.skeleton_np == 1) & (terminal_neighbor_count == 1)


		# Get coordinates (in NumPy array order: z, y, x if 3D; just y, x if 2D)
		terminal_coords = np.argwhere(terminal_points)

		branching_coords_1 = np.argwhere(branching_points_1)
		branching_coords_2 = np.argwhere(branching_points_2)
		branching_coords = np.vstack((branching_coords_1, branching_coords_2))

		# Sample additional points from the skeleton
		sampled_points = []
		# Add terminal and branching points
		sampled_points.extend(terminal_coords)
		sampled_points.extend(branching_coords)

		# Sample additional points from the skeleton
		skeleton_coords = np.argwhere(self.skeleton_np == 1)
		remaining_points = number_of_points - len(sampled_points)

		if remaining_points > 0:
			sampled_indices = np.random.choice(len(skeleton_coords), size=remaining_points, replace=False)
			sampled_points.extend(skeleton_coords[sampled_indices])

		self.points = np.array(sampled_points)

		# self.points = np.vstack((sampled_points, terminal_coords, branching_coords))
		return self.points

	def get_keypoints(self):
		if self.points is not None:
			return self.points

		print("========= SKELETONIZING 2D IMAGE =========" )
		# Binarization
		bin_img = sitk.BinaryThreshold(self.img, lowerThreshold=0, upperThreshold=254, insideValue=1, outsideValue=0)

		# Skeletonization
		skeleton = sitk.BinaryThinning(bin_img)

		# Convert to NumPy for analysis
		self.skeleton_np = sitk.GetArrayFromImage(skeleton).astype(np.uint8)


		# Define 2D 8-neighborhood kernel (or 3D if needed)
		lg_branch_kernel = np.array([[1,1,1,1,1],
									[1,0,0,0,1],
									[1,0,0,0,1],
									[1,0,0,0,1],
									[1,1,1,1,1]])

		# Define 2D 8-neighborhood kernel (or 3D if needed)
		sm_branch_kernel = np.array([[0,1,0],
									[1,0,1],
									[0,1,0]])

		# Define 2D 8-neighborhood kernel (or 3D if needed)
		terminal_kernel = np.array([[1,1,1],
									[1,0,1],
									[1,1,1]])

		# Count neighbors using convolution
		branch_neighbor_count_1 = convolve(self.skeleton_np, sm_branch_kernel, mode='constant', cval=0)
		branch_neighbor_count_2 = convolve(self.skeleton_np, lg_branch_kernel, mode='constant', cval=0)

		# Terminal points: 1 neighbor
		branching_points_1 = (self.skeleton_np == 1) & (branch_neighbor_count_1 >= 3) 
		branching_points_2 = (self.skeleton_np == 1) & (branch_neighbor_count_2 >= 5)

		# Count neighbors using convolution
		terminal_neighbor_count = convolve(self.skeleton_np, terminal_kernel, mode='constant', cval=0)

		# Terminal points: 1 neighbor
		terminal_points = (self.skeleton_np == 1) & (terminal_neighbor_count == 1)


		# Get coordinates (in NumPy array order: z, y, x if 3D; just y, x if 2D)
		terminal_coords = np.argwhere(terminal_points)

		branching_coords_1 = np.argwhere(branching_points_1)
		branching_coords_2 = np.argwhere(branching_points_2)
		branching_coords = np.vstack((branching_coords_1, branching_coords_2))


		self.keypoints = np.vstack((terminal_coords, branching_coords))
		return self.points
	
	def get_graph(self  ):
		if self.graph is not None:
			return self.graph
		
		print("========= SKELETONIZING 2D IMAGE =========" )
		# Binarization
		bin_img = sitk.BinaryThreshold(self.img, lowerThreshold=0, upperThreshold=254, insideValue=1, outsideValue=0)

		# Display the binarized image
		plt.imshow(sitk.GetArrayFromImage(bin_img), cmap='gray')
		plt.title("Binarized Image")
		plt.axis('off')
		plt.show()

		# Skeletonization
		skeleton = sitk.BinaryThinning(bin_img)

		# Convert to NumPy for analysis
		self.skeleton_np = sitk.GetArrayFromImage(skeleton).astype(np.uint8)

		# Plot the skeletonized image
		plt.imshow(self.skeleton_np, cmap='gray')
		plt.title("Skeletonized Image")
		plt.axis('off')
		plt.show()

		# Define 2D 8-neighborhood kernel (or 3D if needed)
		lg_branch_kernel = np.array([[1,1,1,1,1],
									[1,0,0,0,1],
									[1,0,0,0,1],
									[1,0,0,0,1],
									[1,1,1,1,1]])

		# Define 2D 8-neighborhood kernel (or 3D if needed)
		sm_branch_kernel = np.array([[0,1,0],
									[1,0,1],
									[0,1,0]])

		# Define 2D 8-neighborhood kernel (or 3D if needed)
		terminal_kernel = np.array([[1,1,1],
									[1,0,1],
									[1,1,1]])

		# Count neighbors using convolution
		branch_neighbor_count_1 = convolve(self.skeleton_np, sm_branch_kernel, mode='constant', cval=0)
		branch_neighbor_count_2 = convolve(self.skeleton_np, lg_branch_kernel, mode='constant', cval=0)

		# Terminal points: 1 neighbor
		branching_points_1 = (self.skeleton_np == 1) & (branch_neighbor_count_1 >= 3) 
		branching_points_2 = (self.skeleton_np == 1) & (branch_neighbor_count_2 >= 5)

		# Count neighbors using convolution
		terminal_neighbor_count = convolve(self.skeleton_np, terminal_kernel, mode='constant', cval=0)

		# Terminal points: 1 neighbor
		terminal_points = (self.skeleton_np == 1) & (terminal_neighbor_count == 1)


		# Get coordinates (in NumPy array order: z, y, x if 3D; just y, x if 2D)
		terminal_coords = np.argwhere(terminal_points)

		branching_coords_1 = np.argwhere(branching_points_1)
		branching_coords_2 = np.argwhere(branching_points_2)
		branching_coords = np.vstack((branching_coords_1, branching_coords_2))

		skel_graph = nx.Graph()

		#adding all nodes
		for i, coord in enumerate(branching_coords):
			skel_graph.add_node(i, coord=tuple(coord))	
			# print("adding node", i, "at", coord)

		# print("terminal nodes")

		for i, coord in enumerate(terminal_coords):
			skel_graph.add_node(i + len(branching_coords), coord=tuple(coord))
			# print("adding node", i + len(branching_coords), "at", coord)

		# find nearest branch and add connection
		for i, coord in enumerate(terminal_coords):
			connected_index = self.find_nearest_branch(self.skeleton_np, tuple(coord), branching_coords )
			# print("connecting terminal point", len(branching_coords)+i, "to branch point", connected_index)
			skel_graph.add_edge(len(branching_coords)+i, connected_index)
			# for ind in connected_indices:
			# 	skel_graph.add_edge(len(terminal_coords)+i, ind)

		for i, coord in enumerate(branching_coords):
			connected_indices = self.find_nearest_branches(self.skeleton_np, tuple(coord), branching_coords)
			# print("connecting branch point", i, "to branch points", connected_indices)
			for ind in connected_indices:
				skel_graph.add_edge(i, ind)


		self.graph = self.merge_branching_points(skel_graph, tolerance=5)

		print("========= SKELETONIZING 2D IMAGE: COMPLETE =========" )
		return self.graph

	def plot_graph(self):
				
		# Plot the skeletonized map with the graph connections
		plt.imshow(self.skeleton_np, cmap='gray')

		# Plot edges
		for (i, j) in self.graph.edges():
			xi, yi = self.graph.nodes[i]['coord']
			xj, yj = self.graph.nodes[j]['coord']
			plt.plot([yi, yj], [xi, xj], 'r-', linewidth=2)

		# Plot nodes
		for i, data in self.graph.nodes(data=True):
			x, y = data['coord']
			plt.plot(y, x, 'bo', markersize=5)

		plt.axis('off')
		plt.show()


	def plot_points(self):
				
		# Plot the skeletonized map with the graph connections
		plt.imshow(self.skeleton_np, cmap='gray')
		# Plot points
		for point in self.points:
			plt.plot(point[1], point[0], 'go', markersize=5)
		plt.axis('off')
		plt.show()

# if __name__ == "__main__":
# 	# name = "03-a"
# 	image_path = os.path.join(os.getcwd(), "output_data", "03-a_coarse_reg.png")
# 	skel = skeletonizer_2d(image_path,invert=True)
# 	graph = skel.get_graph()
# 	print("graph nodes:", graph.nodes())
# 	print("graph edges:", graph.edges())

# 	skel.plot_graph()
