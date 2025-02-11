import numpy as np
import networkx as nx
import leidenalg as la
import igraph as ig
from networkx.algorithms.community import modularity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')


def load_labels(label_path):
    """
    Load ground truth labels from `.y` file (NumPy format).
    
    Parameters:
    label_path (str): Path to the `.y` label file.
    
    Returns:
    dict: Mapping of node IDs to their true class labels.
    """
    try:
        labels = np.load(label_path, allow_pickle=True)
        label_dict = {str(i): np.argmax(label)
                      for i, label in enumerate(labels)}
        print(f"Loaded ground truth labels from {label_path}")
        return label_dict
    except Exception as e:
        print(f"Warning: Could not load labels from {label_path}: {e}")
        return None


def leiden_algorithm(G, dataset, resolution=0.2):
    """
    Runs Leiden community detection algorithm and returns partition mapping.
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


def compute_nmi_ari(true_labels, detected_partition):
    """
    Compute NMI and ARI scores based on true and detected labels.
    
    Parameters:
    true_labels (dict): Ground truth labels.
    detected_partition (dict): Community assignments from Leiden.
    
    Returns:
    dict: NMI and ARI scores.
    """
    if not true_labels or not detected_partition:
        print("Missing data for NMI & ARI calculation.")
        return None

    common_nodes = list(set(true_labels.keys()) &
                        set(detected_partition.keys()))

    true_labels_list = [true_labels[node] for node in common_nodes]
    pred_labels_list = [detected_partition[node] for node in common_nodes]

    results = {
        "nmi": normalized_mutual_info_score(true_labels_list, pred_labels_list),
        "ari": adjusted_rand_score(true_labels_list, pred_labels_list)
    }

    return results


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
                        src = parts[0].split("\t")[1].replace("paper:", "")
                        dst = parts[1].replace("paper:", "")
                        G.add_edge(src, dst)

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

    label_paths = {
        "citeseer": "./dataset/real-node-label/citeseer/ind.citeseer.y",
        "cora": "./dataset/real-node-label/cora/ind.cora.y",
        "pubmed": "./dataset/real-node-label/pubmed/ind.pubmed.y"
    }

    print("\nRunning Leiden clustering on datasets...\n")
    for dataset_name, dataset_path in datasets.items():
        G = load_graph(dataset_name, dataset_path)
        if G is None:
            continue

        true_labels = load_labels(label_paths.get(dataset_name, ""))

        detected_partition = leiden_algorithm(G, dataset_name)

        if true_labels:
            scores = compute_nmi_ari(true_labels, detected_partition)
            print(f"\nResults for {dataset_name}:")
            print(f"NMI: {scores['nmi']:.4f}")
            print(f"ARI: {scores['ari']:.4f}\n")
