import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities
from sklearn.cluster import SpectralClustering
import numpy as np

# Updated dataset paths
datasets = {
    "dolphin": "./dataset/real-world-datasets/dolphin.gml",
    "les_miserables": "./dataset/real-world-datasets/les_miserables.gml",
    "network_science_coauthorship": "./dataset/real-world-datasets/network_science_coauthorship.gml",
}

# Load GML datasets and fix issues


def load_fixed_gml(file_path):
    graph = nx.Graph()
    node_ids = set()

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    current_node = None
    current_edge = None
    edges = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("node"):
            current_node = {}
        elif stripped.startswith("edge"):
            current_edge = {}
        elif stripped.startswith("id"):
            node_id = int(stripped.split(" ")[-1])
            node_ids.add(node_id)
            current_node["id"] = node_id
        elif stripped.startswith("source"):
            current_edge["source"] = int(stripped.split(" ")[-1])
        elif stripped.startswith("target"):
            current_edge["target"] = int(stripped.split(" ")[-1])
        elif stripped == "]":
            if current_node is not None:
                graph.add_node(current_node["id"])
                current_node = None
            if current_edge is not None:
                if current_edge["source"] in node_ids and current_edge["target"] in node_ids:
                    edges.append(
                        (current_edge["source"], current_edge["target"]))
                current_edge = None

    graph.add_edges_from(edges)
    return graph


# Load real-world graphs
graphs = {name: load_fixed_gml(path) for name, path in datasets.items()}

# Load Davis Southern Women Graph (alternative small real-world dataset)
graphs["davis_southern_women"] = nx.davis_southern_women_graph()

# Function to apply Louvain and Spectral Clustering


def apply_community_detection(graph, name):
    # Louvain method
    louvain_comm = louvain_communities(graph)
    louvain_modularity = nx.algorithms.community.quality.modularity(
        graph, louvain_comm)

    # Spectral clustering (only for smaller graphs)
    spectral_modularity = None
    if len(graph.nodes) <= 5000:
        adj_matrix = nx.to_numpy_array(graph)
        num_clusters = min(10, len(graph.nodes))

        spectral = SpectralClustering(
            n_clusters=num_clusters, affinity='precomputed', random_state=42)
        labels = spectral.fit_predict(adj_matrix)

        spectral_comm = {i: [] for i in range(num_clusters)}
        for node, cluster in zip(graph.nodes(), labels):
            spectral_comm[cluster].append(node)
        spectral_comm = list(spectral_comm.values())

        spectral_modularity = nx.algorithms.community.quality.modularity(
            graph, spectral_comm)

    return {
        "Louvain Modularity": louvain_modularity,
        "Spectral Modularity": spectral_modularity if spectral_modularity is not None else "Skipped (Large Graph)"
    }


# Apply community detection on all datasets
community_results = {name: apply_community_detection(
    graph, name) for name, graph in graphs.items()}

# Convert to DataFrame
df_community_results = pd.DataFrame(community_results).T

# Print results
print(df_community_results)

# Save results to CSV (optional)
df_community_results.to_csv("community_detection_results.csv", index=True)
