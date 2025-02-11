import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms.community import louvain_communities, modularity
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import os

# Updated dataset paths (Ensure filenames are correct)
datasets = {
    "dolphins": "./dataset/real-world-datasets/dolphins.gml",
    "les_miserables": "./dataset/real-world-datasets/les_miserables.gml",
    "network_science_coauthorship": "./dataset/real-world-datasets/network_science_coauthorship.gml",
}

# Function to load GML and fix node/edge issues


def load_fixed_gml(file_path):
    graph = nx.Graph()

    if not os.path.exists(file_path):
        print(f"Error: File not found -> {file_path}")
        return None

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
graphs = {name: load_fixed_gml(
    path) for name, path in datasets.items() if os.path.exists(path)}

# Function to compute Conductance


def compute_conductance(graph, community):
    # Convert generator to list
    boundary_edges = list(nx.edge_boundary(graph, community))
    volume = sum(dict(graph.degree(community)).values())
    return len(boundary_edges) / volume if volume > 0 else 0.0

# Function to apply Louvain and Spectral Clustering, and compute Conductance, NMI, ARI


def apply_community_detection(graph, name):
    if graph is None:
        return {
            "Louvain Modularity": None,
            "NMI": None,
            "ARI": None,
            "Spectral NMI": None,
            "Spectral ARI": None,
            "Conductance": None
        }


    # Louvain method
    louvain_comm = louvain_communities(graph)
    louvain_modularity = modularity(graph, louvain_comm)

    # Compute Conductance (for first community)
    conductance_score = None
    if len(louvain_comm) > 1:
        conductance_score = compute_conductance(graph, louvain_comm[0])

    # Spectral clustering (only for smaller graphs)
    spectral_nmi, spectral_ari = None, None
    if len(graph.nodes) <= 5000:
        adj_matrix = nx.to_numpy_array(graph)
        # Max 10 clusters for spectral
        num_clusters = min(10, len(graph.nodes))

        spectral = SpectralClustering(
            n_clusters=num_clusters, affinity='precomputed', random_state=42)
        labels = spectral.fit_predict(adj_matrix)

        # Compute NMI & ARI (only if meaningful labels exist)
        true_labels = {node: idx for idx, community in enumerate(
            louvain_comm) for node in community}
        true_labels_list = [true_labels[node] for node in graph.nodes()]

        spectral_nmi = normalized_mutual_info_score(true_labels_list, labels)
        spectral_ari = adjusted_rand_score(true_labels_list, labels)

    # Print results
    print(f"Louvain Modularity: {louvain_modularity:.6f}")
    print(f"NMI: {spectral_nmi:.6f}, ARI: {spectral_ari:.6f}, Conductance: {conductance_score if conductance_score else 'N/A'}")

    return {
        "Louvain Modularity": louvain_modularity,
        "NMI": spectral_nmi if spectral_nmi is not None else "Skipped",
        "ARI": spectral_ari if spectral_ari is not None else "Skipped",
        "Conductance": conductance_score if conductance_score is not None else "N/A"
    }


# Apply community detection on all datasets
community_results = {name: apply_community_detection(
    graph, name) for name, graph in graphs.items()}

# Convert results to DataFrame
df_community_results = pd.DataFrame(community_results).T

# Save results to CSV in the **same folder as the script**
csv_filename = "community_detection_results.csv"
df_community_results.to_csv(csv_filename, index=True)

print(f"\nResults saved to {csv_filename}")
