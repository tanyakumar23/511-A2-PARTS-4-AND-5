import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np

# Set up directories.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, '../dataset/real-node-label')
REPORTS_DIR = os.path.join(SCRIPT_DIR, '../reports/question2_part2')
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_graph_node_label(graph_file, label_file):
    """
    Load a graph and node labels from pickled files.
    Assumes the graph file contains a dict-of-lists mapping each node to its neighbors.
    """
    with open(graph_file, 'rb') as gf:
        graph_data = pickle.load(gf)
    with open(label_file, 'rb') as lf:
        labels = pickle.load(lf)
    graph = nx.from_dict_of_lists(graph_data)
    return graph, labels

def process_true_labels(true_labels):
    """
    Convert labels that are NumPy arrays (e.g., one-hot vectors)
    into integer labels.
    """
    if isinstance(true_labels, np.ndarray):
        if len(true_labels.shape) > 1:  # One-hot encoded
            return np.argmax(true_labels, axis=1)
        return true_labels  # Already integer labels
    return true_labels  # Return as-is if it's not a numpy array

def group_nodes_by_partition(partition):
    """
    Group nodes by their community from a partition dictionary.
    The partition is assumed to be a dictionary {node: community_id}.
    This returns a list of communities (each community is a list of nodes).
    """
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)
    return list(communities.values())

def apply_louvain(graph):
    """Apply Louvain community detection to the graph."""
    partition = community_louvain.best_partition(graph)
    return partition

def apply_spectral_clustering(graph, n_clusters):
    """
    Apply Spectral Clustering on the graph.
    
    Converts the graph to an adjacency matrix and performs spectral clustering using
    precomputed affinity.
    Returns a dictionary mapping node to cluster label.
    """
    adj_matrix = nx.to_numpy_array(graph)
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(adj_matrix)
    return {node: labels[i] for i, node in enumerate(graph.nodes())}

def evaluate_topology(graph, communities):
    """
    Evaluate the topology of the graph using modularity.
    
    The communities should be provided as a list of lists of nodes.
    """
    modularity = nx.algorithms.community.modularity(graph, communities)
    return {"modularity": modularity}

def evaluate_label_dependent(true_labels, predicted_labels):
    """
    Evaluate the clustering results using label-based metrics:
      - Normalized Mutual Information (NMI)
      - Adjusted Rand Index (ARI)
    """
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    return {"NMI": nmi, "ARI": ari}

def plot_graph(graph, partition, filename):
    """
    Plot and save the graph with community partitions using an improved visualization.
    """
    plt.figure(figsize=(12, 8))
    
    # Use a better layout algorithm for large graphs
    print(f"Computing layout for {len(graph.nodes())} nodes...")
    pos = nx.spring_layout(graph, k=1/np.sqrt(len(graph.nodes())), 
                          iterations=50, seed=42)
    
    # Create a color map with distinct colors
    unique_communities = sorted(set(partition.values()))
    color_map = plt.cm.tab20(np.linspace(0, 1, len(unique_communities)))
    
    # Draw edges with transparency
    nx.draw_networkx_edges(graph, pos, alpha=0.1, edge_color='gray')
    
    # Draw nodes for each community with different colors
    for idx, comm_id in enumerate(unique_communities):
        node_list = [node for node in graph.nodes() if partition[node] == comm_id]
        nx.draw_networkx_nodes(graph, pos, 
                             nodelist=node_list,
                             node_color=[color_map[idx]],
                             node_size=50)
    
    plt.title(f'Community Detection Results - {len(unique_communities)} communities')
    plt.axis('off')
    plt.savefig(os.path.join(REPORTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Define datasets to process
    datasets = ['cora', 'pubmed', 'citeseer']
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {dataset.upper()} dataset")
        print(f"{'='*50}")
        
        # Load the graph and true labels
        dataset_dir = os.path.join(DATASET_DIR, dataset)
        graph_file = os.path.join(dataset_dir, f'ind.{dataset}.graph')
        label_file = os.path.join(dataset_dir, f'ind.{dataset}.y')
        
        try:
            graph, true_labels = load_graph_node_label(graph_file, label_file)
            
            # Process true labels
            true_labels_processed = process_true_labels(true_labels)
            n_clusters = len(np.unique(true_labels_processed))
            
            print(f"Number of nodes: {graph.number_of_nodes()}")
            print(f"Number of edges: {graph.number_of_edges()}")
            print(f"Number of ground truth communities: {n_clusters}")
            print(f"Number of labeled nodes: {len(true_labels_processed)}")
            
            # Apply Louvain
            print("\nApplying Louvain community detection...")
            louvain_partition = apply_louvain(graph)
            louvain_communities = group_nodes_by_partition(louvain_partition)
            
            # Apply Spectral Clustering
            print("Applying Spectral Clustering...")
            spectral_partition = apply_spectral_clustering(graph, n_clusters)
            spectral_communities = group_nodes_by_partition(spectral_partition)
            
            # Evaluate topology metrics
            print("\nEvaluating topology metrics...")
            louvain_topology = evaluate_topology(graph, louvain_communities)
            spectral_topology = evaluate_topology(graph, spectral_communities)
            
            # Evaluate label-dependent metrics
            labeled_nodes = list(range(len(true_labels_processed)))
            true_labels_ordered = true_labels_processed
            predicted_labels_louvain = [louvain_partition[node] for node in labeled_nodes]
            predicted_labels_spectral = [spectral_partition[node] for node in labeled_nodes]
            
            louvain_label_metrics = evaluate_label_dependent(true_labels_ordered, predicted_labels_louvain)
            spectral_label_metrics = evaluate_label_dependent(true_labels_ordered, predicted_labels_spectral)
            
            # Print results
            print("\nResults:")
            print("\nLouvain Method:")
            print(f"Modularity: {louvain_topology['modularity']:.3f}")
            print(f"NMI: {louvain_label_metrics['NMI']:.3f}")
            print(f"ARI: {louvain_label_metrics['ARI']:.3f}")
            
            print("\nSpectral Clustering:")
            print(f"Modularity: {spectral_topology['modularity']:.3f}")
            print(f"NMI: {spectral_label_metrics['NMI']:.3f}")
            print(f"ARI: {spectral_label_metrics['ARI']:.3f}")
            
            # Visualize results
            plot_graph(graph, louvain_partition, f'{dataset}_louvain.png')
            plot_graph(graph, spectral_partition, f'{dataset}_spectral.png')
            
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue
