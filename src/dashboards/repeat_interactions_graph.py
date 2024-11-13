#!/usr/bin/env python3

import sys
from pathlib import Path
import datetime
from datetime import timedelta

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine
from tqdm import tqdm

# Adjust `PYTHONPATH` to include `src` for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import db_config

# Set up database connection
def get_engine():
    connection_string = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    return create_engine(connection_string)

# Query and load posts and comments
def load_data(engine, start_date, end_date, subreddit):
    posts_query = f"""
    SELECT post_id, author AS post_author, subreddit FROM posts
    WHERE created_utc BETWEEN '{start_date}' AND '{end_date}'
    AND subreddit = '{subreddit}'
    """
    comments_query = f"""
    SELECT post_id, author AS comment_author, parent_id FROM comments
    WHERE created_utc BETWEEN '{start_date}' AND '{end_date}'
    """
    posts = pd.read_sql(posts_query, engine)
    comments = pd.read_sql(comments_query, engine)
    return posts, comments

# Find repeat interactions based on operator and threshold
def find_repeat_interactions(posts, comments, interaction_threshold, operator):
    interactions = comments.merge(posts, on='post_id')
    user_pairs = (
        interactions.groupby(['post_author', 'comment_author'])
        .size()
        .reset_index(name='interaction_count')
    )

    # Apply the operator filter
    if operator == ">=":
        repeat_interactions = user_pairs[user_pairs['interaction_count'] >= interaction_threshold]
    elif operator == "<=":
        repeat_interactions = user_pairs[user_pairs['interaction_count'] <= interaction_threshold]
    elif operator == "=":
        repeat_interactions = user_pairs[user_pairs['interaction_count'] == interaction_threshold]
    else:
        repeat_interactions = pd.DataFrame()  # Empty if no valid operator
    return repeat_interactions

# Calculate the percentage of repeat interacting users in the subreddit
def calculate_intersecting_percentage(repeat_interactions, total_users_in_subreddit):
    unique_repeat_users = pd.concat([repeat_interactions['post_author'], repeat_interactions['comment_author']]).nunique()
    percentage = (unique_repeat_users / total_users_in_subreddit) * 100 if total_users_in_subreddit > 0 else 0
    return percentage

# Build the network graph with progress bar
def create_network(repeat_interactions):
    G = nx.Graph()
    print("Building the network graph...")
    for _, row in tqdm(repeat_interactions.iterrows(), total=len(repeat_interactions), desc="Adding edges"):
        G.add_edge(row['post_author'], row['comment_author'], weight=row['interaction_count'])
    return G

# Create animation frames for each time step
def create_animation_frames(start_date, end_date, time_step, subreddit, interaction_threshold, operator):
    frames = []
    current_start = start_date
    time_step_map = {'Day': timedelta(days=1), 'Week': timedelta(weeks=1), 'Month': timedelta(weeks=4)}

    while current_start < end_date:
        current_end = min(current_start + time_step_map[time_step], end_date)
        engine = get_engine()
        posts, comments = load_data(engine, current_start, current_end, subreddit)
        repeat_interactions = find_repeat_interactions(posts, comments, interaction_threshold, operator)
        G = create_network(repeat_interactions)
        pos = nx.spring_layout(G, seed=42)

        # Capture nodes and edges for this time frame
        edge_x, edge_y = [], []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Append the frame
        frames.append(go.Frame(data=[
            go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.3, color='gray')),
            go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=3, color='blue'), text=list(G.nodes()), hoverinfo='text')
        ], name=str(current_start)))
        
        current_start = current_end

    return frames

# Streamlit dashboard setup
def main():
    st.title("Repeat Interacting Users Analysis")

    # User input for analysis timeframe and subreddit filter
    st.sidebar.header("Select Analysis Filters")
    start_date = st.sidebar.date_input("Start Date", datetime(2024, 10, 1))
    end_date = st.sidebar.date_input("End Date", datetime(2024, 11, 30))

    engine = get_engine()

    # Fetch distinct subreddits for filtering options
    subreddits = pd.read_sql("SELECT DISTINCT subreddit FROM posts", engine)['subreddit'].tolist()
    subreddit = st.sidebar.selectbox("Select Subreddit", subreddits)

    # Add a slider for the repeat interaction threshold and an operator selector
    operator = st.sidebar.selectbox("Select Operator", ["=", ">=", "<="])
    interaction_threshold = st.sidebar.slider("Minimum Repeat Interactions", min_value=1, max_value=25, value=5)

    # Add time-step and animation option
    time_step = st.sidebar.selectbox("Time Step for Animation", ["None", "Day", "Week", "Month"])

    # Load data and calculate statistics for the static view
    posts, comments = load_data(engine, start_date, end_date, subreddit)
    total_users_in_subreddit = pd.concat([posts['post_author'], comments['comment_author']]).nunique()
    st.write(f"Data loaded for subreddit: {subreddit}. Total posts: {len(posts)}, Total comments: {len(comments)}")
    st.write(f"Total users in subreddit: {total_users_in_subreddit}")

    # Identify repeat interactions with the specified threshold and operator
    repeat_interactions = find_repeat_interactions(posts, comments, interaction_threshold, operator)
    st.write(f"Total repeat interactions found with threshold {operator} {interaction_threshold}: {len(repeat_interactions)}")

    # Calculate and display intersecting users percentage
    intersecting_percentage = calculate_intersecting_percentage(repeat_interactions, total_users_in_subreddit)
    st.write(f"Intersecting users in subreddit: {intersecting_percentage:.2f}%")

    # Build the network based on repeat interactions
    G = create_network(repeat_interactions)
    st.write(f"Total nodes (unique users): {G.number_of_nodes()}")
    st.write(f"Total edges (repeat interactions): {G.number_of_edges()}")

    # Default static network visualization
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()

    for edge in G.edges(data=True):
        fig.add_trace(go.Scatter(
            x=[pos[edge[0]][0], pos[edge[1]][0]],
            y=[pos[edge[0]][1], pos[edge[1]][1]],
            mode='lines',
            line=dict(width=0.3, color='gray')
        ))

    for node in G.nodes():
        fig.add_trace(go.Scatter(
            x=[pos[node][0]], y=[pos[node][1]], mode='markers',
            marker=dict(size=3, color='blue'),
            text=node,
            hoverinfo='text'
        ))

    fig.update_layout(
        title="Repeat Interaction Network",
        showlegend=False,
        autosize=False,
        width=800,
        height=600
    )
    st.plotly_chart(fig)

    # Animation only if a time step is selected
    if time_step != "None" and st.sidebar.button("Play Animation"):
        frames = create_animation_frames(start_date, end_date, time_step, subreddit, interaction_threshold, operator)
        
        # Animated visualization
        fig = go.Figure(
            data=[
                go.Scatter(x=[], y=[], mode='lines', line=dict(width=0.3, color='gray')),
                go.Scatter(x=[], y=[], mode='markers', marker=dict(size=3, color='blue'))
            ],
            frames=frames,
            layout=go.Layout(
                title="User Interaction Network Over Time",
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                updatemenus=[{
                    "buttons": [
                        {"args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}], "label": "Play", "method": "animate"},
                        {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                         "label": "Pause", "method": "animate"},
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }],
                sliders=[{
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {"font": {"size": 20}, "prefix": "Date: ", "visible": True, "xanchor": "right"},
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [{"args": [[frame.name], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}],
                               "label": frame.name, "method": "animate"} for frame in frames]
                }]
            )
        )
        st.plotly_chart(fig)

    # Display network data as an exportable table
    st.subheader("Exportable Repeat Interaction Data")
    st.write(repeat_interactions)

    # Provide download link for the data
    csv_data = repeat_interactions.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="repeat_interactions.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

