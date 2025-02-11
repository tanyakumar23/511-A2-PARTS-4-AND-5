import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
import networkx.algorithms.community as nx_community
import time
import pickle
import matplotlib.pyplot as plt


def load_citation_network(network_path, directed=False):
    """
    Load citation network from file
    
    Parameters:
    - network_path: Path to citation network file
    - directed: Whether the network is directed
    
    Returns:
    - NetworkX graph
    """
    # Create graph based on directedness
    G = nx.DiGraph() if directed else nx.Graph()

    try:
        # Different parsing for the different file types
        if network_path.endswith('.cites') or network_path.endswith('.tab'):
            with open(network_path, 'r') as f:
                for line in f:
                    # Skip comments
                    if line.startswith('#') or line.strip() == '':
                        continue

                    # Split line and add edges
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        source, target = parts[0], parts[1]
                        G.add_edge(source, target)
        else:
            raise ValueError(f"Unsupported file format: {network_path}")

        # Convert to undirected for analysis
        G = nx.Graph(G)

        return G

    except Exception as e:
        print(f"Error loading network {network_path}: {e}")
        return None


def load_node_labels(label_path):
    """
    Load node labels from pickle file
    
    Parameters:
    - label_path: Path to node labels file
    
    Returns:
    - Dictionary of node labels
    """
    try:
        with open(label_path, 'rb') as f:
            labels = pickle.load(f, encoding='latin1')

        return labels
    except Exception as e:
        print(f"Error loading labels {label_path}: {e}")
        return None


class ECGClustering:
    def __init__(self, n_clusters=None, n_iterations=10):
        """
        Ensemble Clustering for Graphs (ECG) implementation
        
        Parameters:
        - n_clusters: Number of clusters (if None, will be estimated)
        - n_iterations: Number of ensemble iterations
        """
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations

    def _generate_base_clustering(self, adjacency_matrix):
        """Generate base clustering using different methods"""
        methods = [
            # Spectral Clustering with different parameters
            lambda: SpectralClustering(
                n_clusters=self.n_clusters or 5,
                affinity='precomputed',
                n_init=10,
                random_state=np.random.randint(1000)
            ).fit_predict(adjacency_matrix),

            # Random walk-based community detection
            lambda: self._random_walk_clustering(adjacency_matrix)
        ]

        return methods[np.random.randint(len(methods))]()

    def _random_walk_clustering(self, adjacency_matrix):
        """Implement a random walk-based community detection"""
        G = nx.from_numpy_array(adjacency_matrix)
        communities = list(nx_community.label_propagation_communities(G))

        # Convert community list to labels
        labels = np.zeros(adjacency_matrix.shape[0], dtype=int)
        for i, comm in enumerate(communities):
            for node in comm:
                labels[node] = i

        return labels

    def fit_predict(self, adjacency_matrix):
        """
        Perform Ensemble Clustering for Graphs
        
        Parameters:
        - adjacency_matrix: Numpy array of graph adjacency matrix
        
        Returns:
        - Cluster labels for each node
        """
        adjacency_matrix = np.array(adjacency_matrix)

        # If n_clusters not specified, estimate using spectral clustering
        if self.n_clusters is None:
            self.n_clusters = min(
                max(2, int(np.sqrt(adjacency_matrix.shape[0]))),
                10
            )

        # Ensemble clustering
        ensemble_labels = []
        for _ in range(self.n_iterations):
            ensemble_labels.append(
                self._generate_base_clustering(adjacency_matrix))

        # Convert ensemble to consensus clustering
        ensemble_labels = np.array(ensemble_labels)

        # Voting-based consensus
        final_labels = np.zeros(adjacency_matrix.shape[0], dtype=int)
        for node in range(adjacency_matrix.shape[0]):
            # Get labels for this node across iterations
            node_labels = ensemble_labels[:, node]

            unique_labels, counts = np.unique(node_labels, return_counts=True)
            final_labels[node] = unique_labels[np.argmax(counts)]

        return final_labels


def calculate_modularity(G, labels):
    """
    Calculate modularity for given graph and labels
    
    Parameters:
    - G: NetworkX graph
    - labels: Cluster labels for nodes
    
    Returns:
    - Modularity score
    """
    unique_labels = np.unique(labels)
    communities = [
        [node for node, label in zip(G.nodes(), labels) if label == comm_label]
        for comm_label in unique_labels
    ]

    # Calculate modularity
    return nx_community.modularity(G, communities)


def visualize_graph(G, labels, name):
    """
    Visualize the graph with nodes colored by community
    
    Parameters:
    - G: NetworkX graph
    - labels: Cluster labels for nodes
    - name: Name of the network for the plot title
    """
    plt.figure(figsize=(12, 10))

    # Use spring layout for graph positioning
    pos = nx.spring_layout(G, seed=42)

    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    color_palette = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    # Color nodes based on their community
    node_colors = [color_palette[np.where(unique_labels == label)[
        0][0]] for label in labels]

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=50, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)

    plt.title(f"Community Structure - {name.upper()} Network")
    plt.axis('off')

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{name}_community_graph.png", dpi=300, bbox_inches='tight')
    plt.close()


def process_network(network_path, label_path):
    """
    Process a citation network and calculate modularity
    
    Parameters:
    - network_path: Path to network file
    - label_path: Path to node labels file
    
    Returns:
    - Dictionary of network metrics
    """
    # Load network
    G = load_citation_network(network_path)

    if G is None:
        return None

    # Load node labels for ground truth
    node_labels = load_node_labels(label_path)

    # adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G)

    start_time = time.time()

    # ECG Clustering
    ecg = ECGClustering(n_iterations=10)

    # Perform clustering
    predicted_labels = ecg.fit_predict(adjacency_matrix)

    # Calculate modularity
    modularity = calculate_modularity(G, predicted_labels)

    end_time = time.time()

    results = {
        'Network': network_path.split('/')[-1],
        'Nodes': G.number_of_nodes(),
        'Edges': G.number_of_edges(),
        'Modularity': modularity,
        'Computation Time (s)': end_time - start_time
    }

    # Visualize the graph
    visualize_graph(G, predicted_labels, network_path.split('/')[-1])

    return results


datasets = {
    "citeseer": "./dataset/real-node-label/citeseer/citeseer.cites",
    "cora": "./dataset/real-node-label/cora/cora.cites",
}

label_paths = {
    "citeseer": "./dataset/real-node-label/citeseer/ind.citeseer.y",
    "cora": "./dataset/real-node-label/cora/ind.cora.y",
}

all_results = []
for name, network_path in datasets.items():
    try:
        label_path = label_paths[name]

        result = process_network(network_path, label_path)

        if result:
            print(f"\n{name.upper()} Network Metrics:")
            for key, value in result.items():
                print(f"{key}: {value}")
            all_results.append(result)
    except Exception as e:
        print(f"Error processing {name}: {e}")

for result in all_results:
    print("{:<10} {:<6} {:<6} {:<12.4f} {:<12.4f}".format(
        result['Network'],
        result['Nodes'],
        result['Edges'],
        result['Modularity'],
        result['Computation Time (s)']
    ))
