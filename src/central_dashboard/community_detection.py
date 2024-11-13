#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
import community as community_louvain  # For community detection

# Adjust `PYTHONPATH` to include `src` for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.db_config import db_params
from config.color_palette import COLOR_PALETTE  # Import the centralized color palette

# Set up database connection
def get_engine():
    connection_string = (
        f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}"
        f"@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    )
    return create_engine(connection_string)

# Find user pairs with shared interactions in SQL
def find_user_pairs(engine, start_date, end_date, subreddits, min_shared_interactions):
    params = {
        'start_date': start_date,
        'end_date': end_date,
        'min_shared_interactions': min_shared_interactions
    }
    subreddit_filter = ''
    if subreddits:
        subreddit_list = ', '.join(f"'{subreddit}'" for subreddit in subreddits)
        subreddit_filter = f"AND subreddit IN ({subreddit_list})"

    # Exclude unwanted authors
    author_filter = "AND author != 'None' AND author NOT ILIKE '%Moderator%' AND author != 'autotldr'"

    user_pairs_query = f"""
    WITH interactions AS (
        SELECT author AS username, post_id
        FROM posts
        WHERE created_utc BETWEEN :start_date AND :end_date
        {subreddit_filter}
        {author_filter}
        UNION ALL
        SELECT c.author AS username, c.post_id
        FROM comments c
        JOIN posts p ON c.post_id = p.post_id
        WHERE c.created_utc BETWEEN :start_date AND :end_date
        {subreddit_filter.replace('subreddit', 'p.subreddit')}
        {author_filter.replace('author', 'c.author')}
    ),
    user_posts AS (
        SELECT DISTINCT username, post_id
        FROM interactions
        WHERE post_id IS NOT NULL
    ),
    user_pairs_posts AS (
        SELECT
            up1.username AS user1,
            up2.username AS user2,
            up1.post_id
        FROM user_posts up1
        JOIN user_posts up2 ON up1.post_id = up2.post_id AND up1.username < up2.username
    )
    SELECT
        user1,
        user2,
        COUNT(*) AS shared_interactions
    FROM user_pairs_posts
    GROUP BY user1, user2
    HAVING COUNT(*) >= :min_shared_interactions
    """

    user_pairs_df = pd.read_sql_query(text(user_pairs_query), engine, params=params)

    return user_pairs_df

# Build the network graph
def build_network_graph(user_pairs_df):
    G = nx.Graph()

    # Add edges with weights
    for _, row in user_pairs_df.iterrows():
        G.add_edge(row['user1'], row['user2'], weight=row['shared_interactions'])

    return G

# Visualize the network graph with community colors
def visualize_graph(G, partition):
    # Create custom colorscale from COLOR_PALETTE['categorical']
    num_communities = len(set(partition.values()))
    colors = COLOR_PALETTE['categorical']
    if num_communities > len(colors):
        # Extend the color palette if needed
        colors = colors * (num_communities // len(colors) + 1)
    community_colors = {com: colors[idx] for idx, com in enumerate(set(partition.values()))}

    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}")
        node_color.append(community_colors[partition[node]])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=10,
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='User Interaction Network with Communities',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=800
                    )
                   )
    return fig

# Function to load target subreddits from JSON file
def load_target_subreddits():
    target_subreddits_path = Path(__file__).resolve().parent.parent / "data" / "target_subreddits.json"
    try:
        with open(target_subreddits_path, "r") as file:
            data = json.load(file)
        return data.get("subreddits", [])
    except FileNotFoundError:
        st.error(f"Configuration file not found: {target_subreddits_path}")
        return []
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from file: {target_subreddits_path}")
        return []

# Main Streamlit dashboard function
def main():
    st.set_page_config(layout="wide")
    st.title("User Interaction Network Analysis")

    # Initialize database connection
    engine = get_engine()

    # Sidebar filters
    st.sidebar.header("Select Analysis Filters")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=7))
    end_date = st.sidebar.date_input("End Date", datetime.now())

    if start_date > end_date:
        st.sidebar.error("Error: End date must fall after start date.")
        return

    MAX_DATE_RANGE = timedelta(days=30)
    if (end_date - start_date) > MAX_DATE_RANGE:
        st.sidebar.error(f"Error: Date range cannot exceed {MAX_DATE_RANGE.days} days.")
        return

    # Load target subreddits from JSON file
    subreddits = load_target_subreddits()
    if not subreddits:
        st.sidebar.error("No subreddits available for selection.")
        return

    # Subreddit selection without default
    selected_subreddits = st.sidebar.multiselect("Select Subreddits", subreddits)

    # Minimum shared interactions
    min_shared_interactions = st.sidebar.number_input("Minimum Shared Interactions", min_value=1, value=2)

    # Compute user pairs with shared interactions
    user_pairs_df = find_user_pairs(engine, start_date, end_date, selected_subreddits, min_shared_interactions)

    if user_pairs_df.empty:
        st.write("No user pairs with shared interactions found.")
        return

    # Additional filtering for unwanted usernames in user pairs
    user_pairs_df = user_pairs_df[
        (~user_pairs_df['user1'].str.contains('Moderator', case=False, na=False)) &
        (~user_pairs_df['user2'].str.contains('Moderator', case=False, na=False)) &
        (user_pairs_df['user1'] != 'None') &
        (user_pairs_df['user2'] != 'None') &
        (user_pairs_df['user1'] != 'autotldr') &
        (user_pairs_df['user2'] != 'autotldr')
    ]

    if user_pairs_df.empty:
        st.write("No user pairs after applying filters.")
        return

    # Build the network graph
    G = build_network_graph(user_pairs_df)

    # Detect communities using Louvain method
    partition = community_louvain.best_partition(G)

    # Provide download link for the network graph (edges and nodes)
    edges_df = nx.to_pandas_edgelist(G)
    edges_csv = edges_df.to_csv(index=False)
    nodes_df = pd.DataFrame({'username': list(G.nodes())})
    nodes_df['community'] = nodes_df['username'].map(partition)
    nodes_csv = nodes_df.to_csv(index=False)

    # Output number of clusters and nodes
    st.write(f"### Number of users: {len(nodes_df)}")
    st.write(f"### Unique clusters detected: {len(list(nodes_df['community'].unique()))}")

    # Visualize the graph with communities
    fig = visualize_graph(G, partition)
    st.plotly_chart(fig)

    # Show tables
    st.write("### Network Edges and Nodes Tables")
    st.download_button(
        label="Download Network Edges CSV",
        data=edges_csv,
        file_name="network_edges.csv",
        mime="text/csv"
    )
    st.write(edges_df)
    st.download_button(
        label="Download Network Nodes CSV",
        data=nodes_csv,
        file_name="network_nodes.csv",
        mime="text/csv"
    )
    st.write(nodes_df)


if __name__ == "__main__":
    main()
