import networkx as nx
import leidenalg as la
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.algorithms.community import modularity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')


def visualize_communities(G, partition, dataset_name):
    """
    Visualize the detected communities.
    
    Parameters:
    G (networkx.Graph): The input graph.
    partition (dict): Node to community mapping.
    dataset_name (str): Name of the dataset.
    
    Saves:
    PNG file with a community visualization.
    """
    plt.figure(figsize=(10, 8))

    num_communities = len(set(partition.values()))
    colors = list(mcolors.TABLEAU_COLORS.values())
    if num_communities > len(colors):
        colors = plt.cm.tab20(np.linspace(0, 1, num_communities))

    color_map = [colors[partition[node] % len(colors)] for node in G.nodes()]

    pos = nx.spring_layout(
        G, seed=42, k=1/np.sqrt(G.number_of_nodes()), iterations=50)
    nx.draw(G, pos, node_color=color_map, node_size=50,
            with_labels=False, edge_color="gray", alpha=0.5)

    plt.title(
        f"Leiden Communities for {dataset_name} ({num_communities} communities)")
    plt.savefig(f"{dataset_name}_leiden_communities.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved as {dataset_name}_leiden_communities.png")


def leiden_algorithm(G, dataset, resolution=0.2):
    """
    Runs Leiden community detection algorithm with resolution tuning.
    
    Parameters:
    G (networkx.Graph): Input graph
    dataset (str): Dataset name to apply resolution tuning
    
    Returns:
    dict: Node to community mapping
    """
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    reverse_map = {idx: node for node, idx in node_map.items()}
    edges = [(node_map[u], node_map[v]) for u, v in G.edges()]

    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(node_map))
    ig_graph.add_edges(edges)

    if dataset == "citeseer":
        resolution = 0.02
    elif dataset == "cora":
        resolution = 0.1
    elif dataset == "pubmed":
        resolution = 0.1

    partition = la.find_partition(
        ig_graph,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        n_iterations=10,
        seed=42
    )

    partition_dict = {}
    for idx, comm in enumerate(partition):
        for node_idx in comm:
            original_node = reverse_map[node_idx]
            partition_dict[original_node] = idx

    return partition_dict


def load_graph(dataset_name, dataset_path):
    """
    Loads the dataset properly.
    - Uses `.cites` files for Citeseer and Cora.
    - Parses `Pubmed-Diabetes.DIRECTED.cites.tab` correctly.
    """
    try:
        G = nx.Graph()

        if dataset_name in ['citeseer', 'cora']:
            with open(dataset_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        G.add_edge(str(parts[0]), str(parts[1]))

        elif dataset_name == "pubmed":
            with open(dataset_path, 'r') as f:
                for line in f:
                    if line.startswith("DIRECTED") or line.startswith("NO_FEATURES"):
                        continue

                    parts = line.strip().split("\t|\t")
                    if len(parts) == 2:
                        source = parts[0].split("\t")[1].replace("paper:", "")
                        target = parts[1].replace("paper:", "")
                        G.add_edge(source, target)

        if nx.is_directed(G):
            G = G.to_undirected()

        G.remove_edges_from(nx.selfloop_edges(G))

        print(
            f"Loaded {dataset_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None


if __name__ == "__main__":
    datasets = {
        "citeseer": "./dataset/real-node-label/citeseer/citeseer.cites",
        "cora": "./dataset/real-node-label/cora/cora.cites",
        "pubmed": "./dataset/real-node-label/pubmed/Pubmed-Diabetes.DIRECTED.cites.tab"
    }

    print("\nRunning Leiden clustering on datasets...\n")
    for dataset_name, dataset_path in datasets.items():
        G = load_graph(dataset_name, dataset_path)
        if G is None:
            continue

        partition = leiden_algorithm(G, dataset_name)

        visualize_communities(G, partition, dataset_name)
