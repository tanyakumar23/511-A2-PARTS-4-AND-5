import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import networkx.algorithms.community as nx_community
import time
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


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


def load_network(filepath):
    """
    Load network and handle potential duplicate edges
    
    Parameters:
    - filepath: Path to the GML file
    
    Returns:
    - NetworkX graph with duplicate edges removed
    """
    try:
        # Read the graph
        G = nx.read_gml(filepath)

        # Convert to undirected graph to remove parallel edges
        return nx.Graph(G)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def calculate_metrics(G, labels):
    """
    Calculate various network community metrics
    
    Parameters:
    - G: NetworkX graph
    - labels: Cluster labels for nodes
    
    Returns:
    - Dictionary of metrics
    """
    # Prepare unique communities
    unique_labels = np.unique(labels)
    communities = [
        [node for node, label in zip(G.nodes(), labels) if label == comm_label]
        for comm_label in unique_labels
    ]

    # Modularity
    modularity = nx_community.modularity(G, communities)

    # Conductance (average across communities)
    conductance_values = []
    for comm in communities:
        # Compute internal and external edges
        internal_edges = sum(1 for u in comm for v in comm if G.has_edge(u, v))
        boundary_edges = sum(1 for u in comm for v in G.nodes()
                             if v not in comm and G.has_edge(u, v))

        # Compute conductance (lower is better)
        if internal_edges + boundary_edges > 0:
            conductance_values.append(
                boundary_edges / (internal_edges + boundary_edges))
        else:
            conductance_values.append(0)

    avg_conductance = np.mean(conductance_values)

    # Check for ground truth labels if available
    ground_truth_labels = None
    for node in G.nodes():
        if 'value' in G.nodes[node]:
            ground_truth_labels = [G.nodes[node]['value']
                                   for node in G.nodes()]
            break

    # NMI and ARI
    nmi, ari = 0, 0
    if ground_truth_labels:
        nmi = normalized_mutual_info_score(ground_truth_labels, labels)
        ari = adjusted_rand_score(ground_truth_labels, labels)

    return {
        'Modularity': modularity,
        'Avg Conductance': avg_conductance,
        'NMI': nmi,
        'ARI': ari
    }


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


# Define dataset paths
datasets = {
    "karate": "./dataset/data-subset/karate.gml",
    "polbooks": "./dataset/data-subset/polbooks.gml",
    "strike": "./dataset/data-subset/strike.gml",
}

# Process networks and calculate metrics using ECG
results = {}
for name, path in datasets.items():
    try:
        # Load network
        G = load_network(path)

        if G is None:
            continue

        # Prepare adjacency matrix
        adjacency_matrix = nx.to_numpy_array(G)

        # Measure time and calculate metrics with ECG
        start_time = time.time()

        # Instantiate ECG Clustering
        ecg = ECGClustering(n_iterations=10)

        # Perform clustering
        labels = ecg.fit_predict(adjacency_matrix)

        # Calculate metrics
        metrics = calculate_metrics(G, labels)

        end_time = time.time()

        # Visualize the graph
        visualize_graph(G, labels, name)

        # Store results
        results[name] = {
            'Nodes': G.number_of_nodes(),
            'Edges': G.number_of_edges(),
            **metrics,
            'Computation Time (s)': end_time - start_time
        }

        # Print results for each network
        print(f"\n{name.upper()} Network:")
        for key, value in results[name].items():
            print(f"{key}: {value}")

        print(f"Graph visualization saved as {name}_community_graph.png")

    except FileNotFoundError:
        print(f"File {path} not found. Skipping.")
    except Exception as e:
        print(f"Error processing {name}: {e}")

# Print summary of results
print("\nSUMMARY OF RESULTS:")
for name, result in results.items():
    print(f"{name.upper()}:")
    print(f"  Modularity: {result['Modularity']:.4f}")
    print(f"  Avg Conductance: {result['Avg Conductance']:.4f}")
    print(f"  NMI: {result['NMI']:.4f}")
    print(f"  ARI: {result['ARI']:.4f}")
    print(f"  Computation Time: {result['Computation Time (s)']:.4f}s")
