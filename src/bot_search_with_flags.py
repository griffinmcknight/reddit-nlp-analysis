import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sqlalchemy import create_engine
from db_config import db_params
from tqdm import tqdm
from datetime import datetime, timedelta

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

    # Load flagged user list
    with open("flagged_users.txt", "r") as f:
        flagged_users = set(line.strip() for line in f)

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

    # Identify flagged users in the network
    flagged_in_network = [node for node in G.nodes() if node in flagged_users]
    flagged_count = len(flagged_in_network)
    st.write(f"Flagged Users Found in Network: {flagged_count}")
    st.write(flagged_in_network)

    # Default static network visualization
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()

    # Draw edges
    for edge in G.edges(data=True):
        fig.add_trace(go.Scatter(
            x=[pos[edge[0]][0], pos[edge[1]][0]],
            y=[pos[edge[0]][1], pos[edge[1]][1]],
            mode='lines',
            line=dict(width=0.3, color='gray'),
            hoverinfo='none'
        ))

    # Draw nodes, highlighting flagged users
    for node in G.nodes():
        node_color = 'red' if node in flagged_users else 'blue'
        node_size = 6 if node in flagged_users else 3
        fig.add_trace(go.Scatter(
            x=[pos[node][0]], y=[pos[node][1]],
            mode='markers',
            marker=dict(size=node_size, color=node_color),
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

