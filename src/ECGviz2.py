import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
import networkx.algorithms.community as nx_community
import time
import pickle
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def load_citation_network(network_path, directed=False):
    """ Load citation network from file """
    G = nx.DiGraph() if directed else nx.Graph()

    try:
        if network_path.endswith('.cites') or network_path.endswith('.tab'):
            with open(network_path, 'r') as f:
                for line in f:
                    if line.startswith('#') or line.strip() == '':
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        source, target = parts[0], parts[1]
                        G.add_edge(source, target)
        else:
            raise ValueError(f"Unsupported file format: {network_path}")

        G = nx.Graph(G)  # Convert to undirected for analysis
        return G
    except Exception as e:
        print(f"Error loading network {network_path}: {e}")
        return None


def load_node_labels(label_path):
    """ Load node labels from pickle file """
    try:
        with open(label_path, 'rb') as f:
            labels = pickle.load(f, encoding='latin1')
        return labels
    except Exception as e:
        print(f"Error loading labels {label_path}: {e}")
        return None


def calculate_modularity(G, labels):
    """ Calculate modularity for given graph and labels """
    unique_labels = np.unique(labels)
    communities = [
        [node for node, label in zip(G.nodes(), labels) if label == comm_label]
        for comm_label in unique_labels
    ]
    return nx_community.modularity(G, communities)


def calculate_conductance(G, partition):
    """ Calculate conductance for given partition """
    conductance_vals = []
    for community in partition.values():
        community_set = set(community)
        cut_size = sum(1 for u, v in G.edges(
            community_set) if v not in community_set)
        volume = sum(1 for u in community_set for v in G.neighbors(u))
        conductance_vals.append(cut_size / volume if volume else 0)
    return np.mean(conductance_vals)


class ECGClustering:
    def __init__(self, n_clusters=None, n_iterations=10):
        """ Ensemble Clustering for Graphs (ECG) implementation """
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations

    def fit_predict(self, adjacency_matrix):
        """ Perform Ensemble Clustering for Graphs """
        adjacency_matrix = np.array(adjacency_matrix)
        if self.n_clusters is None:
            self.n_clusters = min(
                max(2, int(np.sqrt(adjacency_matrix.shape[0]))), 10)
        ensemble_labels = []
        for _ in range(self.n_iterations):
            labels = SpectralClustering(
                n_clusters=self.n_clusters, affinity='precomputed').fit_predict(adjacency_matrix)
            ensemble_labels.append(labels)
        ensemble_labels = np.array(ensemble_labels)
        final_labels = np.zeros(adjacency_matrix.shape[0], dtype=int)
        for node in range(adjacency_matrix.shape[0]):
            node_labels = ensemble_labels[:, node]
            unique_labels, counts = np.unique(node_labels, return_counts=True)
            final_labels[node] = unique_labels[np.argmax(counts)]
        return final_labels


def process_network(network_path, label_path):
    """ Process a citation network and calculate metrics """
    G = load_citation_network(network_path)
    if G is None:
        return None

    node_labels = load_node_labels(label_path)
    adjacency_matrix = nx.to_numpy_array(G)
    start_time = time.time()

    ecg = ECGClustering(n_iterations=10)
    predicted_labels = ecg.fit_predict(adjacency_matrix)

    modularity = calculate_modularity(G, predicted_labels)
    conductance = calculate_conductance(G, predicted_labels)

    true_labels = [node_labels[node] for node in G.nodes()]
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)

    end_time = time.time()

    results = {
        'Network': network_path.split('/')[-1],
        'Nodes': G.number_of_nodes(),
        'Edges': G.number_of_edges(),
        'Modularity': modularity,
        'Conductance': conductance,
        'NMI': nmi,
        'ARI': ari,
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
        label_path = label_paths[name]
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
print("{:<10} {:<6} {:<6} {:<12} {:<12} {:<12} {:<12}".format(
    "Network", "Nodes", "Edges", "Modularity", "Conductance", "NMI", "ARI"))
print("-" * 80)

for result in all_results:
    print("{:<10} {:<6} {:<6} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
        result['Network'],
        result['Nodes'],
        result['Edges'],
        result['Modularity'],
        result['Conductance'],
        result['NMI'],
        result['ARI']
    ))
