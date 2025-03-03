import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
import networkx.algorithms.community as nx_community
import time
import pickle


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
        # Different parsing for different file types
        if network_path.endswith('.cites') or network_path.endswith('.tab'):
            with open(network_path, 'r') as f:
                for line in f:
                    # Skip comments and headers
                    if line.startswith('#') or line.strip() == '':
                        continue

                    # Split line and add edge
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
        # Ensure input is numpy array
        adjacency_matrix = np.array(adjacency_matrix)

        # If n_clusters not specified, estimate using spectral clustering
        if self.n_clusters is None:
            # Simple heuristic for estimating number of clusters
            self.n_clusters = min(
                max(2, int(np.sqrt(adjacency_matrix.shape[0]))),
                10  # Cap the number of clusters
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

            # Use mode (most frequent label)
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
    # Create communities from labels
    unique_labels = np.unique(labels)
    communities = [
        [node for node, label in zip(G.nodes(), labels) if label == comm_label]
        for comm_label in unique_labels
    ]

    # Calculate modularity
    return nx_community.modularity(G, communities)


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

    # Prepare adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G)

    # Measure time and calculate modularity with ECG
    start_time = time.time()

    # Instantiate ECG Clustering
    ecg = ECGClustering(n_iterations=10)

    # Perform clustering
    predicted_labels = ecg.fit_predict(adjacency_matrix)

    # Calculate modularity
    modularity = calculate_modularity(G, predicted_labels)

    end_time = time.time()

    # Prepare results
    results = {
        'Network': network_path.split('/')[-1],
        'Nodes': G.number_of_nodes(),
        'Edges': G.number_of_edges(),
        'Modularity': modularity,
        'Computation Time (s)': end_time - start_time
    }

    return results


# Define dataset paths
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

# Process all networks
all_results = []
for name, network_path in datasets.items():
    try:
        # Get corresponding label path
        label_path = label_paths[name]

        # Process network
        result = process_network(network_path, label_path)

        if result:
            print(f"\n{name.upper()} Network Metrics:")
            for key, value in result.items():
                print(f"{key}: {value}")
            all_results.append(result)
    except Exception as e:
        print(f"Error processing {name}: {e}")

# Print summary table
print("\n--- SUMMARY TABLE ---")
print("{:<10} {:<6} {:<6} {:<12} {:<12}".format(
    "Network", "Nodes", "Edges", "Modularity", "Computation Time"))
print("-" * 50)

for result in all_results:
    print("{:<10} {:<6} {:<6} {:<12.4f} {:<12.4f}".format(
        result['Network'],
        result['Nodes'],
        result['Edges'],
        result['Modularity'],
        result['Computation Time (s)']
    ))
