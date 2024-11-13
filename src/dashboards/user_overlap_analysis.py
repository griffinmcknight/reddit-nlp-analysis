import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine
import networkx as nx
import matplotlib.pyplot as plt

# Adjust `PYTHONPATH` to include `src` for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.db_config import db_params

# Set up database connection
def get_engine():
    connection_string = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    return create_engine(connection_string)

# Load subreddits from JSON file
def load_subreddits():
    subreddits_path = Path(__file__).resolve().parent.parent / 'data' / 'target_subreddits.json'
    with open(subreddits_path) as f:
        subreddits = json.load(f).get("subreddits", [])
    return subreddits

# Load posts and comments data with filters applied
def load_data(engine, start_date, end_date, subreddits_selected):
    posts_query = f"""
    SELECT post_id, created_utc, subreddit, title, author, score 
    FROM posts 
    WHERE created_utc BETWEEN '{start_date}' AND '{end_date}' 
    """
    
    if subreddits_selected:
        subreddit_filter = ', '.join([f"'{s}'" for s in subreddits_selected])
        posts_query += f"AND subreddit IN ({subreddit_filter})"
    
    comments_query = f"""
    SELECT post_id, score AS comment_score, author AS commenter_id
    FROM comments
    WHERE created_utc BETWEEN '{start_date}' AND '{end_date}'
    """
    
    if subreddits_selected:
        comments_query += f"AND post_id IN (SELECT post_id FROM posts WHERE subreddit IN ({subreddit_filter}))"
    
    posts = pd.read_sql(posts_query, engine)
    comments = pd.read_sql(comments_query, engine)
    
    return posts, comments

# Display Top N% Posts Table
def get_top_posts(posts_data, top_percentage):
    threshold_score = np.percentile(posts_data['score'], 100 - top_percentage)
    top_posts = posts_data[posts_data['score'] >= threshold_score]
    return top_posts

# Build Network Graph with Progress Bar
def build_network_graph(top_posts, comments_data):
    G = nx.Graph()
    post_ids = top_posts['post_id'].unique()
    
    # Filter comments based on top posts
    top_comments = comments_data[comments_data['post_id'].isin(post_ids)]
    
    # Initialize Streamlit progress bar
    progress_bar = st.progress(0)
    total_posts = len(post_ids)
    
    for idx, post_id in enumerate(post_ids):
        # Find unique commenters per post
        commenters = top_comments[top_comments['post_id'] == post_id]['commenter_id'].unique()
        for i in range(len(commenters)):
            for j in range(i + 1, len(commenters)):
                user1, user2 = commenters[i], commenters[j]
                
                # Add nodes if they don't exist
                if user1 not in G:
                    G.add_node(user1, size=0)
                if user2 not in G:
                    G.add_node(user2, size=0)
                
                # Add edge or update weight
                if G.has_edge(user1, user2):
                    G[user1][user2]['weight'] += 1
                else:
                    G.add_edge(user1, user2, weight=1)
        
        # Update progress bar
        progress_bar.progress((idx + 1) / total_posts)
    
    progress_bar.empty()  # Clear progress bar when complete
    return G

# Plot Network Graph
def plot_network_graph(G):
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    node_sizes = [G.nodes[node]['size'] * 10 for node in G]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges]
    
    plt.figure(figsize=(12, 12))
    nx.draw_networkx(
        G, pos, node_size=node_sizes, width=edge_weights,
        with_labels=False, edge_color='gray', alpha=0.7
    )
    st.pyplot(plt)

# Streamlit Dashboard
def main():
    st.set_page_config(layout="wide")
    st.title("User Overlap Network Graph")

    engine = get_engine()
    all_subreddits = load_subreddits()  # Load subreddits from JSON file
    
    # Sidebar filters
    start_date = st.sidebar.date_input("Start Date", datetime(2024, 10, 1))
    end_date = st.sidebar.date_input("End Date", datetime(2024, 10, 31))
    subreddits_selected = st.sidebar.multiselect("Select Subreddits", options=all_subreddits, default=all_subreddits)
    top_percentage = st.sidebar.slider("Select Top N% of Posts by Score", min_value=5, max_value=100, value=20, step=5)
    
    # Load data
    posts_data, comments_data = load_data(engine, start_date, end_date, subreddits_selected)
    st.metric("Number of Posts", len(posts_data))
    st.metric("Number of Comments", len(comments_data))

    # Get top N% posts and build network graph
    top_posts = get_top_posts(posts_data, top_percentage)
    G = build_network_graph(top_posts, comments_data)
    
    # Display network graph
    plot_network_graph(G)

if __name__ == "__main__":
    main()
