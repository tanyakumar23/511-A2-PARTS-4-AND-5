import os
import re
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
from sklearn.cluster import SpectralClustering

# Set up directories.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, '../dataset/data-subset')
REPORTS_DIR = os.path.join(SCRIPT_DIR, '../reports/question2_part1')
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_graph_classic(filename):
    """
    Manually load a GML file to build an undirected graph while avoiding duplicate edges.
    
    This function:
      1. Reads the file content.
      2. Uses regex to extract node and edge blocks.
      3. For nodes, extracts the id (and optionally a label) and adds them.
      4. For edges, extracts the source and target, skips duplicates, and adds the edge.
    
    This approach avoids issues with duplicate edge definitions (as seen in football.gml)
    and ensures the graph is undirected.
    """
    with open(filename, 'r') as f:
        file_content = f.read()
    
    # Extract all node blocks.
    node_blocks = re.findall(r'node\s*\[\s*(.*?)\s*\]', file_content, re.DOTALL)
    # Extract all edge blocks.
    edge_blocks = re.findall(r'edge\s*\[\s*(.*?)\s*\]', file_content, re.DOTALL)
    
    G = nx.Graph()
    
    # Process node blocks.
    for block in node_blocks:
        id_match = re.search(r'id\s+("?\S+"?)', block)
        if not id_match:
            continue
        node_id = id_match.group(1).strip('"')
        label_match = re.search(r'label\s+"(.*?)"', block)
        label = label_match.group(1) if label_match else node_id
        G.add_node(node_id, label=label)
    
    # Process edge blocks.
    seen_edges = set()
    for block in edge_blocks:
        src_match = re.search(r'source\s+("?\S+"?)', block)
        tgt_match = re.search(r'target\s+("?\S+"?)', block)
        if not (src_match and tgt_match):
            continue
        src = src_match.group(1).strip('"')
        tgt = tgt_match.group(1).strip('"')
        # For undirected graphs, order the pair so (a, b) and (b, a) are the same.
        edge_key = tuple(sorted((src, tgt)))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        G.add_edge(src, tgt)
    
    if len(G) == 0:
        raise ValueError(f"Empty graph loaded from {filename}")
    return G

def evaluate_topology(graph, communities):
    modularity = nx.algorithms.community.modularity(graph, communities)
    return {"modularity": modularity}

def group_nodes_by_partition(partition):
    """Group nodes by their community ID."""
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)
    return list(communities.values())

def plot_graph(graph, partition, filename):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)
    # Build color list for nodes in the order of graph.nodes()
    colors = [partition[node] for node in graph.nodes()]
    nx.draw(graph, pos, node_color=colors, with_labels=True, node_size=500, cmap='viridis')
    plt.title('Graph Community Visualization')
    plt.savefig(os.path.join(REPORTS_DIR, filename))
    plt.close()

def apply_louvain(graph):
    """Apply Louvain community detection to the full graph."""
    partition = community_louvain.best_partition(graph)
    return partition

def apply_spectral_clustering(graph, n_clusters):
    """
    Apply Spectral Clustering on a graph.
    
    If the graph is disconnected, the clustering is performed on its largest connected component.
    In that case, if the requested number of clusters exceeds the number of nodes in that component,
    n_clusters is reduced accordingly.
    
    Returns a tuple (partition, subgraph) where partition is a dictionary mapping each node in the
    subgraph to a cluster label.
    """
    if not nx.is_connected(graph):
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc).copy()
    else:
        subgraph = graph
    
    n_samples = subgraph.number_of_nodes()
    if n_clusters > n_samples:
        n_clusters = n_samples

    adj_matrix = nx.to_numpy_array(subgraph)
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(adj_matrix)
    partition = {node: labels[i] for i, node in enumerate(subgraph.nodes())}
    return partition, subgraph

# List of graph filenames to process.
classic_graphs = ['strike.gml', 'karate.gml', 'polblogs.gml', 'polbooks.gml', 'football.gml']

for graph_file in classic_graphs:
    try:
        print(f"\nProcessing {graph_file}...")
        graph_path = os.path.join(DATASET_DIR, graph_file)
        graph = load_graph_classic(graph_path)

        # Louvain community detection on the full graph.
        print(f"Applying Louvain to {graph_file}...")
        louvain_partition = apply_louvain(graph)
        plot_graph(graph, louvain_partition, f"{graph_file}_louvain.png")
        communities = group_nodes_by_partition(louvain_partition)
        topology_eval = evaluate_topology(graph, communities)
        print(f"Louvain on {graph_file}: {topology_eval}")
        
        # Use the number of communities from Louvain for spectral clustering.
        n_clusters = len(set(louvain_partition.values()))
        print(f"Applying Spectral Clustering to {graph_file} with {n_clusters} clusters...")
        spectral_partition, spectral_graph = apply_spectral_clustering(graph, n_clusters)
        plot_graph(spectral_graph, spectral_partition, f"{graph_file}_spectral.png")
        spectral_communities = group_nodes_by_partition(spectral_partition)
        spectral_eval = evaluate_topology(spectral_graph, spectral_communities)
        print(f"Spectral Clustering on {graph_file}: {spectral_eval}")
        
    except Exception as e:
        print(f"Error processing {graph_file}: {str(e)}")
        continue
