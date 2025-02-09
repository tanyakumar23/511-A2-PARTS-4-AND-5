import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle 
from community import community_louvain
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '../dataset')
REPORTS_DIR = os.path.join(SCRIPT_DIR, '../reports/question2')

os.makedirs(REPORTS_DIR, exist_ok=True)

def load_graph_classic(filename):
    """Load a classic graph from GML file, handling duplicate edges"""
    try:
        # Read the file as a MultiGraph first to handle duplicate edges
        G = nx.read_gml(filename, label='id')  # Use 'id' as node label
        # Convert to simple graph, combining parallel edges
        simple_G = nx.Graph()
        for u, v, data in G.edges(data=True):
            if simple_G.has_edge(u, v):
                # If edge exists, update weight if it exists
                simple_G[u][v]['weight'] = simple_G[u][v].get('weight', 1) + 1
            else:
                # Add new edge
                simple_G.add_edge(u, v, **data)
        return simple_G
    except Exception as e:
        print(f"Warning: Error loading {filename}: {str(e)}")
        return None

def load_graph_node_label(graph_file, label_file):
    """Load a graph and its labels from pickle files"""
    try:
        with open(graph_file, 'rb') as file:
            graph_data = pickle.load(file)
        with open(label_file, 'rb') as file:
            labels = pickle.load(file)
            # Convert 2D labels to 1D by taking argmax
            if len(labels.shape) > 1:
                labels = labels.argmax(axis=1)
        graph = nx.Graph(nx.from_dict_of_lists(graph_data))
        return graph, labels
    except Exception as e:
        print(f"Warning: Error loading {graph_file}: {str(e)}")
        return None, None

def evaluate_topology(graph, communities):
    """Evaluate community structure using modularity"""
    try:
        modularity = nx.algorithms.community.modularity(graph, communities)
        return {"modularity": modularity}
    except Exception as e:
        print(f"Warning: Error calculating modularity: {str(e)}")
        return {"modularity": None}

def get_communities_from_partition(partition):
    """Convert a partition dictionary into a list of community lists"""
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)
    return list(communities.values())

def evaluate_label_dependency(true_labels, predicted_labels):
    """Evaluate clustering against ground truth labels"""
    try:
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels)
        return {"nmi": nmi, "ari": ari}
    except Exception as e:
        print(f"Warning: Error calculating metrics: {str(e)}")
        return {"nmi": None, "ari": None}

def plot_graph(graph, partition, filename):
    """Plot graph with community colors"""
    try:
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(graph, seed=42)
        color = [partition[node] for node in graph.nodes()]
        nx.draw_networkx(graph, pos, node_color=color, with_labels=True, 
                        node_size=500, cmap='viridis')
        plt.title('Graph Community Visualization')
        clean_filename = filename.replace('/', '_')
        plt.savefig(os.path.join(REPORTS_DIR, clean_filename))
        plt.close()
    except Exception as e:
        print(f"Warning: Error plotting graph: {str(e)}")

def apply_louvain(graph):
    """Apply Louvain community detection"""
    try:
        return community_louvain.best_partition(graph)
    except Exception as e:
        print(f"Warning: Error in Louvain algorithm: {str(e)}")
        return None

def apply_spectral(graph):
    """Apply Spectral Clustering"""
    try:
        # Estimate number of communities using eigenvalues
        laplacian = nx.normalized_laplacian_matrix(graph)
        n_clusters = min(len(graph), 8)  # limit to reasonable number
        
        adjacency_matrix = nx.to_numpy_array(graph)
        spectral = SpectralClustering(n_clusters=n_clusters, 
                                    affinity='precomputed',
                                    random_state=42)
        labels = spectral.fit_predict(adjacency_matrix)
        return {node: labels[i] for i, node in enumerate(graph.nodes())}
    except Exception as e:
        print(f"Warning: Error in Spectral Clustering: {str(e)}")
        return None

# Define datasets
classic_graphs = [
    os.path.join('data-subset', 'strike.gml'),
    os.path.join('data-subset', 'karate.gml'),
    os.path.join('data-subset', 'polblogs.gml'),
    os.path.join('data-subset', 'polbooks.gml'),
    os.path.join('data-subset', 'football.gml')
]

real_node_labels = [
    ('cora/ind.cora.graph', 'cora/ind.cora.y'),
    ('citeseer/ind.citeseer.graph', 'citeseer/ind.citeseer.y'),
    ('pubmed/ind.pubmed.graph', 'pubmed/ind.pubmed.y')
]

def main():
    print("\n=== Classic Graphs Analysis ===")
    for graph_file in classic_graphs:
        print(f"\nAnalyzing {graph_file}")
        graph_path = os.path.join(DATA_DIR, graph_file)
        graph = load_graph_classic(graph_path)
        if graph is None:
            continue

        # Apply and evaluate Louvain
        louvain_partition = apply_louvain(graph)
        if louvain_partition:
            base_filename = os.path.basename(graph_file)
            plot_graph(graph, louvain_partition, f'louvain_{base_filename}.png')
            communities = get_communities_from_partition(louvain_partition)
            topology_eval = evaluate_topology(graph, communities)
            print(f"Louvain modularity: {topology_eval}")

        # Apply and evaluate Spectral Clustering
        spectral_partition = apply_spectral(graph)
        if spectral_partition:
            plot_graph(graph, spectral_partition, f"spectral_{base_filename}.png")
            communities = get_communities_from_partition(spectral_partition)
            topology_eval = evaluate_topology(graph, communities)
            print(f"Spectral modularity: {topology_eval}")

    print("\n=== Real Graphs with Labels Analysis ===")
    for graph_file, label_file in real_node_labels:
        print(f"\nAnalyzing {graph_file}")
        graph_path = os.path.join(DATA_DIR, 'real-node-label', graph_file)
        label_path = os.path.join(DATA_DIR, 'real-node-label', label_file)

        graph, true_labels = load_graph_node_label(graph_path, label_path)
        if graph is None:
            continue

        # Apply and evaluate both algorithms
        for algorithm, func in [("Louvain", apply_louvain), 
                              ("Spectral", apply_spectral)]:
            partition = func(graph)
            if partition:
                plot_graph(graph, partition, f"{graph_file}_{algorithm}.png")
                communities = get_communities_from_partition(partition)
                topology_eval = evaluate_topology(graph, communities)
                print(f"{algorithm} modularity: {topology_eval}")
                
                # Compare with ground truth labels
                predicted_labels = list(partition.values())
                label_eval = evaluate_label_dependency(true_labels, predicted_labels)
                print(f"{algorithm} label comparison: {label_eval}")

if __name__ == "__main__":
    main()

    
