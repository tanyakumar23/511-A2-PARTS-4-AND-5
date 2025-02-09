import networkx as nx
import pandas as pd
import os 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMAIL_DATA_DIR = os.path.join(SCRIPT_DIR, '../dataset', 'email-Enron.txt')
ADDRESS_DATA_DIR = os.path.join(SCRIPT_DIR, '../dataset', 'addresses-email-Enron.txt')

email_data = pd.read_csv(EMAIL_DATA_DIR, sep=' ', header=None, names=['sender', 'recipient', 'timestamp'])
adresses_data = pd.read_csv(ADDRESS_DATA_DIR, sep='\t', header=None, names=['id', 'email'])

G = nx.Graph()

edge_weights = email_data.groupby(['sender', 'recipient']).size().reset_index(name='weight')

for index, row in edge_weights.iterrows():
    G.add_edge(row['sender'], row['recipient'], weight=row['weight'])

email_map = dict(zip(adresses_data['id'].astype(int), adresses_data['email']))

# Add this after creating email_map to debug
print("Email map keys:", list(email_map.keys())[:5])
print("Email map sample:", dict(list(email_map.items())[:5]))

degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

def get_centrality_scores(centrality_dict, n=5):
    sorted_centrality = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True) [:n]
    return sorted_centrality

top_degree = get_centrality_scores(degree_centrality, 5)
top_betweenness = get_centrality_scores(betweenness_centrality, 5)
top_closeness = get_centrality_scores(closeness_centrality, 5)
top_eigenvector = get_centrality_scores(eigenvector_centrality, 5)


def write_centrality_report():
    """Write centrality measurements to a markdown report file"""
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(SCRIPT_DIR, '../reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    report_path = os.path.join(reports_dir, 'centrality_measured.md')
    
    with open(report_path, 'w') as f:
        f.write("# Network Centrality Analysis\n\n")
        
        f.write("## Top 5 Nodes by Degree Centrality\n")
        for node, score in top_degree:
            print(f"Node type: {type(node)}")
            print(f"Node value: {node}")
            print(f"email_map keys type: {type(list(email_map.keys())[0])}")
            print(f"Available keys: {list(email_map.keys())[:5]}")  # Print first 5 keys
            f.write(f"- {email_map[int(node)]}: {score:.4f}\n")
            
        f.write("\n## Top 5 Nodes by Betweenness Centrality\n")
        for node, score in top_betweenness:
            print(f"Node type: {type(node)}")
            print(f"Node value: {node}")
            print(f"email_map keys type: {type(list(email_map.keys())[0])}")
            print(f"Available keys: {list(email_map.keys())[:5]}")  # Print first 5 keys
            f.write(f"- {email_map[int(node)]}: {score:.4f}\n")
            
        f.write("\n## Top 5 Nodes by Closeness Centrality\n")
        for node, score in top_closeness:
            print(f"Node type: {type(node)}")
            print(f"Node value: {node}")
            print(f"email_map keys type: {type(list(email_map.keys())[0])}")
            print(f"Available keys: {list(email_map.keys())[:5]}")  # Print first 5 keys
            f.write(f"- {email_map[int(node)]}: {score:.4f}\n")
            
        f.write("\n## Top 5 Nodes by Eigenvector Centrality\n")
        for node, score in top_eigenvector:
            print(f"Node type: {type(node)}")
            print(f"Node value: {node}")
            print(f"email_map keys type: {type(list(email_map.keys())[0])}")
            print(f"Available keys: {list(email_map.keys())[:5]}")  # Print first 5 keys
            f.write(f"- {email_map[int(node)]}: {score:.4f}\n")

if __name__ == "__main__":
    write_centrality_report()




