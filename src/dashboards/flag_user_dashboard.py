#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# Adjust `PYTHONPATH` to include `src` for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.db_config import db_params

# Set up database connection
def get_engine():
    connection_string = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    return create_engine(connection_string)

# Function to remove duplicates in the 'flagged_user_interactions' table
def remove_duplicates(engine):
    with engine.connect() as conn:
        delete_duplicates_query = text("""
        DELETE FROM flagged_user_interactions
        WHERE id NOT IN (
            SELECT MIN(id) 
            FROM flagged_user_interactions
            GROUP BY username, subreddit, post_id, created_utc, interaction_type
        );
        """)
        conn.execute(delete_duplicates_query)
        print("Duplicate entries removed from flagged_user_interactions table.")

# Load interactions for flagged users with optional filters
def load_flagged_user_interactions(engine, start_date, end_date, subreddit=None):
    query = f"""
    SELECT username, subreddit, post_id, interaction_type, created_utc
    FROM flagged_user_interactions
    WHERE created_utc BETWEEN '{start_date}' AND '{end_date}'
    """
    if subreddit:
        query += f" AND subreddit = '{subreddit}'"
    
    interactions = pd.read_sql(query, engine)
    return interactions

# Load post details for posts in flagged user interactions
def load_post_details(engine, post_ids):
    if len(post_ids) == 0:
        return pd.DataFrame(columns=["post_id", "author", "title", "created_utc", "score"])

    query = f"""
    SELECT post_id, author, title, created_utc, score
    FROM posts
    WHERE post_id IN ({','.join([f"'{pid}'" for pid in post_ids])})
    """
    post_details = pd.read_sql(query, engine)
    return post_details

# Build the network graph
def create_network(interactions):
    G = nx.Graph()
    for _, row in interactions.iterrows():
        if row['interaction_type'] == 'comment':
            G.add_edge(row['username'], row['post_id'])
    return G

# Main Streamlit dashboard function
def main():
    st.title("Flagged User Interaction Analysis")

    # Initialize database connection and remove duplicates
    engine = get_engine()
    remove_duplicates(engine)

    # Sidebar filters
    st.sidebar.header("Select Analysis Filters")
    start_date = st.sidebar.date_input("Start Date", datetime(2024, 10, 1))
    end_date = st.sidebar.date_input("End Date", datetime(2024, 11, 30))

    # Fetch distinct subreddits for filtering options
    subreddits = pd.read_sql("SELECT DISTINCT subreddit FROM flagged_user_interactions", engine)['subreddit'].tolist()
    subreddit = st.sidebar.selectbox("Optional Subreddit Filter", ["All"] + subreddits)
    subreddit = None if subreddit == "All" else subreddit

    # Load interactions with filters applied
    interactions = load_flagged_user_interactions(engine, start_date, end_date, subreddit)
    st.write(f"Total interactions loaded: {len(interactions)}")

    # Network visualization of interactions among flagged users
    G = create_network(interactions)
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()

    # Add edges
    for edge in G.edges():
        fig.add_trace(go.Scatter(
            x=[pos[edge[0]][0], pos[edge[1]][0]],
            y=[pos[edge[0]][1], pos[edge[1]][1]],
            mode='lines',
            line=dict(width=0.3, color='gray')
        ))

    # Add nodes with hover labels as usernames
    for node in G.nodes():
        # Check if the node is in the interactions as a post or comment
        post_row = interactions[(interactions['post_id'] == node) & (interactions['interaction_type'] == 'post')]
        comment_row = interactions[(interactions['post_id'] == node) & (interactions['interaction_type'] == 'comment')]

        if not post_row.empty:
            # If the node corresponds to a post, use the post's author as the label
            node_label = post_row['username'].values[0]
            color = 'blue'
            size = 3
        elif not comment_row.empty:
            # If the node corresponds to a comment, use the comment's author as the label
            node_label = comment_row['username'].values[0]
            color = 'green'
            size = 3
        else:
            # Fallback if the node doesn't match any post or comment
            node_label = "Unknown"
            color = 'gray'
            size = 3

        # If the node corresponds to a flagged user, make it red and larger
        if node_label in interactions['username'].values:
            color = 'red'
            size = 6

        # Add the node to the graph plot
        fig.add_trace(go.Scatter(
            x=[pos[node][0]],
            y=[pos[node][1]],
            mode='markers',
            marker=dict(size=size, color=color),
            text=node_label,
            hoverinfo='text'
        ))


    fig.update_layout(
        title="Flagged User Interaction Network",
        showlegend=False,
        autosize=False,
        width=800,
        height=600
    )
    st.plotly_chart(fig)

    # Display flagged user interactions by subreddit
    st.subheader("Flagged User Interaction Counts by Subreddit")
    subreddit_counts = interactions['subreddit'].value_counts().reset_index()
    subreddit_counts.columns = ['Subreddit', 'Interaction Count']
    st.write(subreddit_counts)

    # Load and display details of posts in the network graph
    post_ids = interactions['post_id'].unique()
    post_details = load_post_details(engine, post_ids)

    st.subheader("Details of Posts in the Network")
    st.write(post_details)

    # Display interaction data as an exportable table
    st.subheader("Exportable Flagged User Interaction Data")
    st.write(interactions)

    # Provide download link for the data
    csv_data = interactions.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="flagged_user_interactions.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

