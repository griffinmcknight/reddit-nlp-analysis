import networkx as nx
import pandas as pd
from cdlib import algorithms
from sqlalchemy import create_engine
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import db_config
import warnings
import numpy as np

# Enhanced warning suppression for optional libraries and 'graph_tool'
warnings.filterwarnings("ignore", category=UserWarning, message="to be able to use all crisp methods")
warnings.filterwarnings("ignore", message="Note: to be able to use all crisp methods, you need to install some additional packages:  {'graph_tool'}")

# Connect to database and load data
def get_engine():
    params = db_config.db_params
    connection_string = f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['dbname']}"
    return create_engine(connection_string)

# Function to process a batch of comments
def process_comment_batch(batch, comments_data):
    edges = []
    for _, comment in batch.iterrows():
        parent_author = comments_data[comments_data['comment_id'] == comment['parent_id']]['author'].values
        if parent_author.size > 0:
            edges.append((parent_author[0], comment['author']))
    return edges

def main():
    # Step 1: Load Data
    print("Step 1: Loading data from database...")
    engine = get_engine()
    posts_data = pd.read_sql("SELECT post_id, subreddit, author FROM posts", engine)
    comments_data = pd.read_sql("SELECT comment_id, post_id, parent_id, author FROM comments", engine)
    engine.dispose()
    print("Data loaded successfully. Total posts:", len(posts_data), "Total comments:", len(comments_data))

    # Step 2: Set User Interaction Threshold
    threshold = 10  # Set this to 1 for batch-only, or higher to filter by frequent users

    # Step 3: Filter by Frequent Users if Threshold > 1
    if threshold > 1:
        print(f"\nStep 3: Filtering for users with {threshold}+ comments...")
        frequent_users = comments_data['author'].value_counts()
        frequent_users = frequent_users[frequent_users >= threshold].index
        filtered_comments = comments_data[comments_data['author'].isin(frequent_users)]
        print(f"Filtered to {len(filtered_comments)} comments from {len(frequent_users)} frequent users.")
    else:
        print("\nStep 3: Skipping user filtering and proceeding with batch processing of all comments.")
        filtered_comments = comments_data  # No filtering, use all comments

    # Step 4: Create User Interaction Network with Batched Parallel Processing
    print("\nStep 4: Creating user interaction network with parallel edge creation in batches...")
    G = nx.DiGraph()

    # Define batch size
    batch_size = 5000  # Tune based on available memory and performance
    comment_batches = np.array_split(filtered_comments, len(filtered_comments) // batch_size)

    # Process each batch in parallel
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_comment_batch, batch, filtered_comments) for batch in comment_batches]
        with tqdm(total=len(futures), desc="Processing comment batches") as pbar:
            for future in as_completed(futures):
                edges = future.result()
                if edges:
                    G.add_edges_from(edges)  # Add all edges from the batch to the graph
                pbar.update(1)

    print("User interaction network created successfully.")
    print(f"Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")

    # Step 5: Community Detection using Leiden Algorithm
    print("\nStep 5: Detecting communities using Leiden algorithm...")
    communities = algorithms.leiden(G.to_undirected())
    partition = {node: community_id for community_id, community in enumerate(communities.communities) for node in community}

    # Assign communities as node attributes and print community count
    nx.set_node_attributes(G, partition, 'community')
    num_communities = len(set(partition.values()))
    print(f"Community detection completed. Number of communities detected: {num_communities}")

    # Step 6: Save Nodes and Edges to CSV for Streamlit
    print("\nStep 6: Saving nodes and edges to CSV files for Streamlit visualization...")
    nodes = pd.DataFrame([{'user': node, 'community': data['community']} for node, data in G.nodes(data=True)])
    edges = pd.DataFrame([{'source': u, 'target': v} for u, v in G.edges()])

    # Save nodes and edges with progress updates
    nodes_file = 'network_nodes.csv'
    edges_file = 'network_edges.csv'
    nodes.to_csv(nodes_file, index=False)
    print(f"Nodes saved to {nodes_file}. Total nodes: {len(nodes)}")
    edges.to_csv(edges_file, index=False)
    print(f"Edges saved to {edges_file}. Total edges: {len(edges)}")

    print("\nNetwork analysis completed successfully.")

# Ensure script runs with proper multiprocessing context
if __name__ == '__main__':
    main()
