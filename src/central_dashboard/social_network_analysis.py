#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

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

# Load aggregated interactions from the database
def load_interactions(engine, start_date, end_date, subreddits):
    params = {'start_date': start_date, 'end_date': end_date}
    subreddit_filter = ''
    if subreddits:
        subreddit_list = ', '.join(f"'{subreddit}'" for subreddit in subreddits)
        subreddit_filter = f"AND subreddit IN ({subreddit_list})"

    # Aggregate interactions from posts and comments
    interactions_query = f"""
    WITH post_interactions AS (
        SELECT
            author AS username,
            COUNT(*) AS interaction_count,
            SUM(score) AS total_score
        FROM posts
        WHERE created_utc BETWEEN :start_date AND :end_date
        {subreddit_filter}
        GROUP BY author
    ),
    comment_interactions AS (
        SELECT
            c.author AS username,
            COUNT(*) AS interaction_count,
            SUM(c.score) AS total_score
        FROM comments c
        JOIN posts p ON c.post_id = p.post_id
        WHERE c.created_utc BETWEEN :start_date AND :end_date
        {subreddit_filter.replace('subreddit', 'p.subreddit')}
        GROUP BY c.author
    )
    SELECT
        username,
        SUM(interaction_count) AS total_interactions,
        SUM(total_score) AS total_score
    FROM (
        SELECT * FROM post_interactions
        UNION ALL
        SELECT * FROM comment_interactions
    ) AS combined_interactions
    GROUP BY username
    """

    interactions = pd.read_sql_query(text(interactions_query), engine, params=params)

    return interactions

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

    user_pairs_query = f"""
    WITH interactions AS (
        SELECT author AS username, post_id
        FROM posts
        WHERE created_utc BETWEEN :start_date AND :end_date
        {subreddit_filter}
        UNION ALL
        SELECT c.author AS username, c.post_id
        FROM comments c
        JOIN posts p ON c.post_id = p.post_id
        WHERE c.created_utc BETWEEN :start_date AND :end_date
        {subreddit_filter.replace('subreddit', 'p.subreddit')}
    ),
    user_posts AS (
        SELECT DISTINCT username, post_id
        FROM interactions
        WHERE post_id IS NOT NULL
    )
    SELECT
        up1.username AS user1,
        up2.username AS user2,
        COUNT(*) AS shared_interactions
    FROM user_posts up1
    JOIN user_posts up2 ON up1.post_id = up2.post_id AND up1.username < up2.username
    GROUP BY up1.username, up2.username
    HAVING COUNT(*) >= :min_shared_interactions
    """

    user_pairs_df = pd.read_sql_query(text(user_pairs_query), engine, params=params)

    return user_pairs_df

# Build the network graph
def build_network_graph(user_pairs_df, user_scores):
    G = nx.Graph()

    # Add nodes with user scores
    for _, row in user_scores.iterrows():
        G.add_node(row['username'], user_score=row['total_score'])

    # Add edges with weights
    for _, row in user_pairs_df.iterrows():
        G.add_edge(row['user1'], row['user2'], weight=row['shared_interactions'])

    return G

# Visualize the network graph with gradient colorization
def visualize_graph(G):
    # Create custom colorscale from COLOR_PALETTE['continuous']
    num_colors = len(COLOR_PALETTE['continuous'])
    custom_colorscale = []
    for i, color in enumerate(COLOR_PALETTE['continuous']):
        custom_colorscale.append((i / (num_colors - 1), color))

    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    edge_weights = []
    edge_texts = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = edge[2]['weight']
        edge_weights.append(weight)
        edge_texts.append(f"{edge[0]} ↔ {edge[1]}: {weight} shared posts")

    # Normalize edge weights for color mapping
    max_weight = max(edge_weights) if edge_weights else 1
    min_weight = min(edge_weights) if edge_weights else 0
    weight_range = max_weight - min_weight if max_weight != min_weight else 1
    edge_colors = [(w - min_weight) / weight_range for w in edge_weights]

    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1),
        hoverinfo='text',
        mode='lines',
        text=edge_texts,
        marker=dict(
            color=edge_colors,
            colorscale=custom_colorscale,
            showscale=False
        )
    )

    node_x = []
    node_y = []
    node_scores = []
    node_text = []
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_scores.append(node[1]['user_score'])
        node_text.append(f"{node[0]}<br>User Score: {node[1]['user_score']}")

    # Normalize node scores for color mapping
    max_score = max(node_scores) if node_scores else 1
    min_score = min(node_scores) if node_scores else 0
    score_range = max_score - min_score if max_score != min_score else 1
    node_colors = [(score - min_score) / score_range for score in node_scores]

    # Normalize node sizes for visualization
    node_sizes_normalized = [10 + (score / max_score) * 40 if max_score > 0 else 10 for score in node_scores]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale=custom_colorscale,
            color=node_colors,
            size=node_sizes_normalized,
            colorbar=dict(
                thickness=15,
                title='User Score',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='User Interaction Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper"
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor=COLOR_PALETTE["background"]
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

    # Load aggregated interactions
    interactions = load_interactions(engine, start_date, end_date, selected_subreddits)

    if interactions.empty:
        st.write("No interactions found for the selected filters.")
        return

    st.write(f"Total users with interactions: {len(interactions)}")

    # Apply minimum user score filter
    MIN_USER_SCORE = st.sidebar.number_input("Minimum User Score", min_value=0, value=10)
    interactions = interactions[interactions['total_score'] >= MIN_USER_SCORE]

    if interactions.empty:
        st.write("No users meet the minimum score requirement.")
        return

    # Compute user pairs with shared interactions
    user_pairs_df = find_user_pairs(engine, start_date, end_date, selected_subreddits, min_shared_interactions)

    if user_pairs_df.empty:
        st.write("No user pairs with shared interactions found.")
        return

    # Filter user pairs to include only users in interactions
    user_pairs_df = user_pairs_df[
        user_pairs_df['user1'].isin(interactions['username']) &
        user_pairs_df['user2'].isin(interactions['username'])
    ]

    if user_pairs_df.empty:
        st.write("No user pairs after applying user score filters.")
        return

    st.subheader("User Pairs with Shared Interactions")
    st.write(user_pairs_df)

    # Build the network graph
    G = build_network_graph(user_pairs_df, interactions)

    # Visualize the graph
    fig = visualize_graph(G)
    st.plotly_chart(fig)

    # Additional data display
    st.subheader("User Scores")
    st.write(interactions.sort_values(by='total_score', ascending=False))

    # Provide download link for the data
    csv_data = interactions.to_csv(index=False)
    st.download_button(
        label="Download User Interactions CSV",
        data=csv_data,
        file_name="user_interactions.csv",
        mime="text/csv"
    )

    user_pairs_csv = user_pairs_df.to_csv(index=False)
    st.download_button(
        label="Download User Pairs CSV",
        data=user_pairs_csv,
        file_name="user_pairs.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
