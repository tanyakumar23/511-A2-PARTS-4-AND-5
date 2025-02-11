import networkx as nx
import leidenalg as la
import igraph as ig
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from networkx.algorithms.community import modularity
import warnings
import time
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')


def leiden_algorithm(G, dataset_name):
    """
    Implementation of Leiden community detection algorithm with progress tracking
    """
    print(f"\nStarting Leiden algorithm for {dataset_name}")
    start_time = time.time()

    try:
        # Create node mapping for consistent node indices
        print("Creating node mapping...")
        node_map = {node: idx for idx, node in enumerate(G.nodes())}
        reverse_map = {idx: node for node, idx in node_map.items()}

        # Convert edges using the mapping
        print("Converting edges...")
        edges = [(node_map[u], node_map[v]) for u, v in G.edges()]

        # Create igraph object with progress updates
        print("Creating igraph object...")
        ig_graph = ig.Graph()
        ig_graph.add_vertices(len(node_map))
        print(f"Added {len(node_map)} vertices")

        # Add edges in batches to show progress
        batch_size = 10000
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i+batch_size]
            ig_graph.add_edges(batch)
            print(
                f"Added edges: {min(i+batch_size, len(edges))}/{len(edges)}", end='\r')
        print("\nFinished adding edges")

        # Run Leiden with progress updates
        print("Running Leiden algorithm...")
        partition = la.find_partition(
            ig_graph,
            la.ModularityVertexPartition,
            n_iterations=10,
            seed=42
        )

        # Convert result back to original node labels with progress updates
        print("Converting results back to original labels...")
        partition_dict = {}
        total_nodes = sum(len(comm) for comm in partition)
        nodes_processed = 0

        for idx, comm in enumerate(partition):
            for node_idx in comm:
                original_node = reverse_map[node_idx]
                partition_dict[original_node] = idx
                nodes_processed += 1
                if nodes_processed % 1000 == 0:
                    print(
                        f"Processed nodes: {nodes_processed}/{total_nodes}", end='\r')

        elapsed_time = time.time() - start_time
        print(f"\nLeiden algorithm completed in {elapsed_time:.2f} seconds")
        return partition_dict

    except Exception as e:
        print(f"\nError: {str(e)}")
        return None


def calculate_conductance(G, communities):
    """Calculate conductance for each community and return average"""
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


def evaluate_clustering(G, partition, dataset_name):
    """Evaluate clustering using multiple metrics"""
    communities = {}
    for node, cluster in partition.items():
        if cluster not in communities:
            communities[cluster] = []
        communities[cluster].append(node)
    community_list = list(communities.values())

    results = {
        "modularity": modularity(G, community_list),
        "conductance": calculate_conductance(G, community_list),
        "num_communities": len(community_list)
    }

    return results


def load_graph(dataset_name, dataset_path):
    """Load graph with sampling for large datasets"""
    print(f"\nLoading {dataset_name}...")
    start_time = time.time()

    try:
        if dataset_name in ['citeseer', 'cora', 'pubmed']:
            print("Reading binary/text file...")
            try:
                with open(dataset_path, 'rb') as f:
                    content = f.read()
                print(f"File size: {len(content)/1024:.1f} KB")
                edges = []
                offset = 0
                total_size = len(content)
                last_percent = -1

                while offset < len(content) - 8:
                    current_percent = int((offset / total_size) * 100)
                    if current_percent > last_percent:
                        print(f"Processing file: {current_percent}%", end='\r')
                        last_percent = current_percent

                    try:
                        node1 = int.from_bytes(
                            content[offset:offset+4], byteorder='little')
                        node2 = int.from_bytes(
                            content[offset+4:offset+8], byteorder='little')
                        edges.append((str(node1), str(node2)))
                        offset += 8
                    except:
                        offset += 1
                print("\nFile processing complete.")
                G = nx.Graph(edges)

                # Apply sampling only for Pubmed
                if dataset_name == 'pubmed':
                    sample_size = int(G.number_of_nodes()
                                      * 0.3)  # 30% sampling
                    print(
                        f"\nSampling {sample_size} nodes from {G.number_of_nodes()} total nodes for Pubmed...")
                    sampled_nodes = np.random.choice(list(G.nodes()),
                                                     size=sample_size,
                                                     replace=False)
                    G = G.subgraph(sampled_nodes)
                    print(
                        f"Created sampled graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

                # Apply sampling for Pubmed
                if dataset_name == 'pubmed':
                    sample_size = int(G.number_of_nodes()
                                      * 0.3)  # 30% sampling
                    print(
                        f"\nSampling {sample_size} nodes from {G.number_of_nodes()} total nodes...")
                    sampled_nodes = np.random.choice(list(G.nodes()),
                                                     size=sample_size,
                                                     replace=False)
                    G = G.subgraph(sampled_nodes)
                    print(
                        f"Created sampled graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

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
            # For other datasets - try GML format
            try:
                G = nx.read_gml(dataset_path)
            except:
                try:
                    G = nx.read_gml(dataset_path, label="id")
                except:
                    # Final attempt with edge duplicates removed
                    G = nx.Graph(nx.read_gml(dataset_path, label="id"))

        # Ensure no parallel edges by converting to simple graph
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G

    except Exception as e:
        print(f"Error loading {dataset_name}: {str(e)}")
        return None


def visualize_communities(G, partition, output_path=None):
    """Visualize communities with different colors"""
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


def run_leiden_evaluation(datasets):
    """Run Leiden algorithm on multiple datasets and evaluate results"""
    results = {}
    total_start_time = time.time()

    for dataset_name, dataset_path in datasets.items():
        dataset_start_time = time.time()
        print(f"\nProcessing {dataset_name}...")
        try:
            # Load graph
            G = load_graph(dataset_name, dataset_path)
            if G is None:
                continue

            print(
                f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            print(
                f"Estimated memory usage: {G.size(weight=None) * 16 / (1024*1024):.1f} MB")

            # Calculate estimated time based on graph size
            # rough estimate in seconds
            estimated_time = (G.number_of_nodes() *
                              G.number_of_edges()) / 1000000
            print(
                f"Estimated processing time: {timedelta(seconds=int(estimated_time))}")

            # Calculate Leiden communities
            leiden_partition = leiden_algorithm(G, dataset_name)
            if leiden_partition is None:
                continue

            # Evaluate results
            print("Evaluating clustering results...")
            metrics = evaluate_clustering(G, leiden_partition, dataset_name)

            # Print results
            print(f"\nResults for {dataset_name}:")
            print(f"Modularity: {metrics['modularity']:.4f}")
            print(f"Conductance: {metrics['conductance']:.4f}")
            print(f"Number of communities: {metrics['num_communities']}")

            # Generate visualization
            print("Generating visualization...")
            try:
                output_path = f"{dataset_name}_leiden_communities.png"
                visualize_communities(G, leiden_partition, output_path)
                print(f"Visualization saved as {output_path}")
            except Exception as e:
                print(f"Error generating visualization: {e}")

            # Track processing time
            dataset_time = time.time() - dataset_start_time
            metrics['processing_time'] = dataset_time
            print(
                f"\nTotal processing time for {dataset_name}: {dataset_time:.2f} seconds")

            results[dataset_name] = metrics

        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            continue

    # Print total processing time
    total_time = time.time() - total_start_time
    print(
        f"\nTotal processing time for all datasets: {total_time:.2f} seconds")

    return results


if __name__ == "__main__":
    # Define all datasets
    datasets = {
        "karate": "./dataset/data-subset/karate.gml",
        "polbooks": "./dataset/data-subset/polbooks.gml",
        "football": "./dataset/data-subset/football.gml",
        "strike": "./dataset/data-subset/strike.gml",
        "citeseer": "./dataset/real-node-label/citeseer/ind.citeseer.graph",
        "cora": "./dataset/real-node-label/cora/ind.cora.graph",
        "pubmed": "./dataset/real-node-label/pubmed/ind.pubmed.graph"
    }

    # Run evaluation
    results = run_leiden_evaluation(datasets)

    # Print summary
    print("\nSummary of results:")
    for dataset, metrics in results.items():
        print(f"\n{dataset}:")
        if dataset == 'pubmed':
            print("(Using 30% sample of the graph)")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
