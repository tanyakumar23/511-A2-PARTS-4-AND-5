import networkx as nx
import leidenalg as la
import igraph as ig
import numpy as np
from networkx.algorithms.community import modularity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')


def leiden_algorithm(G, dataset, resolution=0.2):
    """
    Runs Leiden community detection algorithm with resolution tuning.
    
    Parameters:
    G (networkx.Graph): Input graph
    dataset (str): Dataset name to apply resolution tuning
    
    Returns:
    dict: Node to community mapping
    """
    # Map nodes to indices
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    reverse_map = {idx: node for node, idx in node_map.items()}

    # Convert edges
    edges = [(node_map[u], node_map[v]) for u, v in G.edges()]

    # Create igraph object
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(node_map))
    ig_graph.add_edges(edges)

    # Adjust resolution based on dataset
    if dataset == "citeseer":
        resolution = 0.02  # Reduced to prevent over-segmentation
    elif dataset == "cora":
        resolution = 0.1   # Increased for better grouping
    elif dataset == "pubmed":
        resolution = 0.1

    # Run Leiden
    partition = la.find_partition(
        ig_graph,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        n_iterations=10,
        seed=42
    )

    # Convert results
    partition_dict = {}
    for idx, comm in enumerate(partition):
        for node_idx in comm:
            original_node = reverse_map[node_idx]
            partition_dict[original_node] = idx

    return partition_dict


def calculate_conductance(G, partition):
    """
    Calculate conductance correctly.
    
    Parameters:
    G (networkx.Graph): Input graph
    partition (dict): Node to community mapping
    
    Returns:
    float: Average conductance value
    """
    community_nodes = {}
    for node, comm in partition.items():
        if comm not in community_nodes:
            community_nodes[comm] = set()
        community_nodes[comm].add(node)

    conductances = []
    for community in community_nodes.values():
        internal_edges = sum(1 for u, v in G.edges(
            community) if v in community)
        external_edges = sum(1 for u, v in G.edges(
            community) if v not in community)

        if internal_edges + external_edges > 0:
            conductance = external_edges / \
                (2 * internal_edges + external_edges)
            conductances.append(conductance)

    return np.mean(conductances) if conductances else 0


def evaluate_clustering(G, partition, dataset_name, true_labels=None):
    """
    Evaluate clustering using modularity, conductance, and NMI/ARI.
    
    Parameters:
    G (networkx.Graph): Input graph
    partition (dict): Node to community mapping
    dataset_name (str): Name of dataset
    true_labels (dict, optional): Ground truth labels
    
    Returns:
    dict: Evaluation metrics
    """
    communities = {}
    for node, cluster in partition.items():
        if cluster not in communities:
            communities[cluster] = []
        communities[cluster].append(node)
    community_list = list(communities.values())

    results = {
        "modularity": modularity(G, community_list),
        "conductance": calculate_conductance(G, partition),
        "num_communities": len(community_list)
    }

    # If ground truth labels are available, compute NMI and ARI
    if true_labels is not None:
        pred_labels = [partition[node] for node in G.nodes()]
        true_labels_list = [true_labels[node] for node in G.nodes()]

        results.update({
            "nmi": normalized_mutual_info_score(true_labels_list, pred_labels),
            "ari": adjusted_rand_score(true_labels_list, pred_labels)
        })

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
            # Load Citeseer and Cora as edge lists
            with open(dataset_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        G.add_edge(str(parts[0]), str(parts[1]))

        elif dataset_name == "pubmed":
            # Parse Pubmed-Diabetes.DIRECTED.cites.tab
            with open(dataset_path, 'r') as f:
                for line in f:
                    if line.startswith("DIRECTED") or line.startswith("NO_FEATURES"):
                        continue  # Skip metadata lines

                    parts = line.strip().split("\t|\t")  # Pubmed format uses `\t|\t` as a separator
                    if len(parts) == 2:
                        source = parts[0].split("\t")[1].replace("paper:", "")
                        target = parts[1].replace("paper:", "")
                        G.add_edge(source, target)

        # Convert to undirected if needed
        if nx.is_directed(G):
            G = G.to_undirected()

        # Remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G))

        print(
            f"Loaded {dataset_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None


def run_leiden_evaluation(datasets):
    """
    Run Leiden algorithm on multiple datasets and evaluate results.
    
    Parameters:
    datasets (dict): Dictionary mapping dataset names to file paths
    
    Returns:
    dict: Results for each dataset
    """
    results = {}

    for dataset_name, dataset_path in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        try:
            G = load_graph(dataset_name, dataset_path)
            if G is None:
                continue

            print(
                f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

            # Run Leiden algorithm
            partition = leiden_algorithm(G, dataset_name)

            # Evaluate results
            metrics = evaluate_clustering(G, partition, dataset_name)

            print(f"Results for {dataset_name}:")
            print(f"Modularity: {metrics['modularity']:.4f}")
            print(f"Conductance: {metrics['conductance']:.4f}")
            print(f"Number of communities: {metrics['num_communities']}")
            if "nmi" in metrics:
                print(f"NMI: {metrics['nmi']:.4f}")
                print(f"ARI: {metrics['ari']:.4f}")

            results[dataset_name] = metrics

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue

    return results


if __name__ == "__main__":
    datasets = {
        "citeseer": "./dataset/real-node-label/citeseer/citeseer.cites",
        "cora": "./dataset/real-node-label/cora/cora.cites",
        "pubmed": "./dataset/real-node-label/pubmed/Pubmed-Diabetes.DIRECTED.cites.tab"
    }

    print("\nRunning Leiden clustering on datasets...")
    results = run_leiden_evaluation(datasets)

    print("\nFinal Results Summary:")
    for dataset, metrics in results.items():
        print(f"\n{dataset}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}" if isinstance(
                value, float) else f"  {metric}: {value}")
