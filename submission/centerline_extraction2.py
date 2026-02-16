import os
import trimesh
import skeletor as sk
import sys
from trimesh.smoothing import filter_laplacian
import networkx as nx

# -------------------------
# 1. Iterate Through Meshes
# -------------------------
# Define the range of mesh names
mesh_prefixes = [f"{i:02d}" for i in range(1, 11)]  # 01, 02, ..., 10
mesh_suffixes = ["a", "b"]  # a, b
mesh_dir = os.path.join(os.getcwd(), "meshes")
graphs_dir = os.path.join(os.getcwd(), "graphs")
os.makedirs(graphs_dir, exist_ok=True)

for prefix in mesh_prefixes:
    for suffix in mesh_suffixes:
        meshname = f"{prefix}-{suffix}.stl"
        mesh_path = os.path.join(mesh_dir, meshname)

        if not os.path.exists(mesh_path):
            print(f"Mesh file not found: {mesh_path}. Skipping...")
            continue

        print(f"Processing mesh: {meshname}")

        # -------------------------
        # 2. Load the Vessel Mesh
        # -------------------------
        try:
            mesh = trimesh.load(mesh_path)
            mesh.apply_translation(-mesh.bounding_box.centroid)
            print("Mesh centroid:", mesh.bounding_box.centroid)

        except Exception as e:
            print(f"Error loading mesh {meshname}: {e}. Skipping...")
            continue

        # Optional: Verify that the mesh is triangulated
        if not all(len(face) == 3 for face in mesh.faces):
            mesh = mesh.triangulate()

        # -------------------------
        # 3. Preprocess the Mesh
        # -------------------------
        fixed_mesh = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
        fixed_mesh = filter_laplacian(fixed_mesh, lamb=0.5, iterations=10)
        contracted_mesh = sk.pre.contract(fixed_mesh, epsilon=0.05)

        # -------------------------
        # 4. Extract the Skeleton
        # -------------------------
        try:
            skeleton = sk.skeletonize.by_teasar(
                contracted_mesh, root=None, inv_dist=3.0, min_length=15
            )
            sk.post.clean_up(skeleton, inplace=True)
            sk.post.smooth(skeleton, inplace=True)
            sk.post.radii(skeleton, method="knn")
        except Exception as e:
            print(f"Error extracting skeleton for {meshname}: {e}. Skipping...")
            continue

        # -------------------------
        # 5. Export the Skeleton Graph
        # -------------------------
        try:
            vertices = {i: {'x': coord[0], 'y': coord[1], 'z': coord[2]} for i, coord in enumerate(skeleton.vertices)}
            edges = [(edge[0], edge[1]) for edge in skeleton.edges]

            G = nx.Graph()
            G.add_nodes_from(vertices.items())
            G.add_edges_from(edges)

            # G = skeleton.get_graph()
            for node, attributes in G.nodes(data=True):
                print(f"Node {node}: {attributes}")


            output_name = f"{os.path.splitext(meshname)[0]}_skeleton_graph.graphml"
            output_path = os.path.join(graphs_dir, output_name)
            nx.write_graphml(G, output_path)
            print(f"Graph exported to {output_path}")
        except Exception as e:
            print(f"Error exporting graph for {meshname}: {e}. Skipping...")
            continue

print("Processing complete.")
