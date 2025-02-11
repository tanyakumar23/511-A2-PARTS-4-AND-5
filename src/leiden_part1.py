import networkx as nx
import leidenalg as la
import igraph as ig
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from networkx.algorithms.community import modularity
import warnings
warnings.filterwarnings('ignore')


def leiden_algorithm(G):
    """
    Implementation of Leiden community detection algorithm
    
    Parameters:
    G (networkx.Graph): Input graph
    
    Returns:
    dict: Node to community mapping
    """
    # Create node mapping for consistent node indices
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    reverse_map = {idx: node for node, idx in node_map.items()}

    # Convert edges using the mapping
    edges = [(node_map[u], node_map[v]) for u, v in G.edges()]

    # Create igraph object
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(node_map))
    ig_graph.add_edges(edges)

    # Run Leiden with resolution parameter optimization
    partition = la.find_partition(
        ig_graph,
        la.ModularityVertexPartition,
        n_iterations=10,
        seed=42
    )

    # Convert result back to original node labels
    partition_dict = {}
    for idx, comm in enumerate(partition):
        for node_idx in comm:
            original_node = reverse_map[node_idx]
            partition_dict[original_node] = idx

    return partition_dict


def calculate_conductance(G, communities):
    """
    Calculate conductance for each community and return average
    """
    conductances = []
    for community in communities:
        internal_edges = G.subgraph(community).number_of_edges()
        external_edges = sum(1 for u, v in G.edges() if
                             (u in community and v not in community) or
                             (v in community and u not in community))
        if internal_edges + external_edges > 0:
            conductance = external_edges / \
                (2 * internal_edges + external_edges)
            conductances.append(conductance)
    return np.mean(conductances) if conductances else 0


def visualize_communities(G, partition, output_path=None):
    """
    Visualize communities with different colors and save the plot
    
    Parameters:
    G (networkx.Graph): Input graph
    partition (dict): Node to community mapping
    output_path (str): Path to save the visualization (optional)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Generate colors for each community
    num_communities = len(set(partition.values()))
    colors = list(mcolors.TABLEAU_COLORS.values())
    if num_communities > len(colors):
        colors = plt.cm.tab20(np.linspace(0, 1, num_communities))

    # Create a color map for nodes based on their community
    color_map = [colors[partition[node] % len(colors)] for node in G.nodes()]

    # Calculate layout
    pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), iterations=50)

    # Draw the network
    nx.draw(G, pos,
            node_color=color_map,
            node_size=100,
            width=0.5,
            with_labels=False,
            edge_color='gray',
            alpha=0.7)

    # Add title
    plt.title(
        f'Community Detection Results\n{G.number_of_nodes()} nodes, {len(set(partition.values()))} communities')

    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def evaluate_clustering(G, partition, dataset_name, true_labels=None):
    """
    Evaluate clustering using multiple metrics
    
    Parameters:
    G (networkx.Graph): Input graph
    partition (dict): Node to community mapping
    dataset_name (str): Name of the dataset
    true_labels (dict, optional): Ground truth labels
    
    Returns:
    dict: Dictionary of evaluation metrics
    """
    # Convert partition to list of communities
    communities = {}
    for node, cluster in partition.items():
        if cluster not in communities:
            communities[cluster] = []
        communities[cluster].append(node)
    community_list = list(communities.values())

    # Calculate metrics
    results = {
        "modularity": modularity(G, community_list),
        "conductance": calculate_conductance(G, community_list),
        "num_communities": len(community_list)
    }

    # Add label-dependent metrics if ground truth available
    if true_labels is not None:
        # Ensure same node ordering
        pred_labels = [partition[node] for node in G.nodes()]
        true_labels_list = [true_labels[node] for node in G.nodes()]

        results.update({
            "nmi": normalized_mutual_info_score(true_labels_list, pred_labels),
            "ari": adjusted_rand_score(true_labels_list, pred_labels)
        })

    # Generate visualization
    try:
        output_path = f"{dataset_name}_leiden_communities.png"
        visualize_communities(G, partition, output_path)
        print(f"Visualization saved as {output_path}")
    except Exception as e:
        print(f"Error generating visualization: {e}")

    return results


def load_graph(dataset_name, dataset_path):
    """
    Load graph with proper format handling for different file types
    """
    try:
        if dataset_name in ['citeseer', 'cora', 'pubmed']:
            # Read binary file
            try:
                with open(dataset_path, 'rb') as f:
                    content = f.read()
                # Try to interpret as adjacency matrix or edge list
                edges = []
                # Skip header bytes and read edge pairs
                offset = 0
                # Need at least 8 bytes for an edge pair
                while offset < len(content) - 8:
                    try:
                        node1 = int.from_bytes(
                            content[offset:offset+4], byteorder='little')
                        node2 = int.from_bytes(
                            content[offset+4:offset+8], byteorder='little')
                        edges.append((str(node1), str(node2)))
                        offset += 8
                    except:
                        offset += 1
                G = nx.Graph(edges)
            except Exception as e:
                print(f"Binary file reading failed: {str(e)}")
                # Fallback: try reading as text
                with open(dataset_path, 'r', encoding='latin1') as f:
                    edges = []
                    for line in f:
                        try:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                edges.append((str(parts[0]), str(parts[1])))
                        except:
                            continue
                G = nx.Graph(edges)
        else:
            # Special handling for football dataset
            if dataset_name == 'football':
                try:
                    # Read the GML file as text first
                    edges = set()  # Use a set to automatically remove duplicates
                    nodes = set()
                    with open(dataset_path, 'r') as f:
                        for line in f:
                            if 'source' in line:
                                source = line.split()[1]
                                nodes.add(source)
                            elif 'target' in line:
                                target = line.split()[1]
                                nodes.add(target)
                                # Add edge in a consistent order to avoid duplicates
                                edge = tuple(sorted([source, target]))
                                edges.add(edge)

                    # Create graph from unique edges
                    G = nx.Graph()
                    G.add_nodes_from(nodes)
                    G.add_edges_from(edges)
                except Exception as e:
                    print(f"Error in football dataset parsing: {e}")
                    raise
            else:
                # For other datasets - try GML format
                try:
                    G = nx.read_gml(dataset_path)
                except:
                    G = nx.read_gml(dataset_path, label="id")

        # Convert to undirected if needed
        if nx.is_directed(G):
            G = G.to_undirected()

        # Remove self-loops and convert node labels to strings
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.relabel_nodes(G, str)

        return G
    except Exception as e:
        print(f"Error loading {dataset_name}: {str(e)}")
        return None


def run_leiden_evaluation(datasets, skip_large_datasets=True):
    """
    Run Leiden algorithm on multiple datasets and evaluate results
    
    Parameters:
    datasets (dict): Dictionary mapping dataset names to file paths
    skip_large_datasets (bool): If True, skip citeseer, cora, and pubmed
    
    Returns:
    dict: Results for each dataset
    """
    results = {}

    for dataset_name, dataset_path in datasets.items():
        # Skip large datasets if requested
        if skip_large_datasets and dataset_name in ['citeseer', 'cora', 'pubmed']:
            print(f"\nSkipping {dataset_name} (large dataset)")
            continue

        print(f"\nProcessing {dataset_name}...")
        try:
            # Load graph
            G = load_graph(dataset_name, dataset_path)
            if G is None:
                continue

            print(
                f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

            # Run Leiden algorithm
            partition = leiden_algorithm(G)

            # Evaluate results
            metrics = evaluate_clustering(G, partition, dataset_name)

            print(f"Results for {dataset_name}:")
            print(f"Modularity: {metrics['modularity']:.4f}")
            print(f"Conductance: {metrics['conductance']:.4f}")
            print(f"Number of communities: {metrics['num_communities']}")

            # Generate visualization
            try:
                output_path = f"{dataset_name}_leiden_communities.png"
                visualize_communities(G, partition, output_path)
                print(f"Visualization saved as {output_path}")
            except Exception as e:
                print(f"Error generating visualization: {e}")

            results[dataset_name] = metrics

        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            continue

    return results


if __name__ == "__main__":
    # Define datasets
    small_datasets = {
        "karate": "./dataset/data-subset/karate.gml",
        "polbooks": "./dataset/data-subset/polbooks.gml",
        "football": "./dataset/data-subset/football.gml",
        "strike": "./dataset/data-subset/strike.gml",
    }

    large_datasets = {
        "citeseer": "./dataset/real-node-label/citeseer/ind.citeseer.graph",
        "cora": "./dataset/real-node-label/cora/ind.cora.graph",
        "pubmed": "./dataset/real-node-label/pubmed/ind.pubmed.graph"
    }

    # First run small datasets
    print("Processing small datasets...")
    results_small = run_leiden_evaluation(small_datasets)

    
    results = results_small

    # Print summary
    print("\nSummary of results:")
    for dataset, metrics in results.items():
        print(f"\n{dataset}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

# Original datasets definition for reference:
datasets = {
    "karate": "./dataset/data-subset/karate.gml",
    "polbooks": "./dataset/data-subset/polbooks.gml",
    "football": "./dataset/data-subset/football.gml",
    "strike": "./dataset/data-subset/strike.gml",
   
}

# Run evaluation
results = run_leiden_evaluation(datasets)
