import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Adjust `PYTHONPATH` to include `src` for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.db_config import db_params
from config.color_palette import COLOR_PALETTE  # Adjust colors as necessary for a cohesive style

# Check for MPS device for Apple Silicon
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Set up database connection
def get_engine():
    connection_string = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    return create_engine(connection_string)

# Load subreddits from target_subreddits.json
def load_subreddits():
    file_path = Path(__file__).resolve().parent.parent / 'data' / 'target_subreddits.json'
    with open(file_path, 'r') as file:
        subreddits_data = json.load(file)
    return subreddits_data.get('subreddits', [])

# Query and load posts and comments
def load_data(engine, start_date, end_date, subreddits_selected):
    posts_query = f"""
    SELECT post_id, created_utc, subreddit, title, author, score 
    FROM posts 
    WHERE created_utc BETWEEN '{start_date}' AND '{end_date}' 
    AND subreddit IN ({', '.join([f"'{s}'" for s in subreddits_selected])})
    """
    comments_query = f"""
    SELECT comments.post_id, comments.created_utc 
    FROM comments 
    JOIN posts ON comments.post_id = posts.post_id 
    WHERE comments.created_utc BETWEEN '{start_date}' AND '{end_date}'
    AND posts.subreddit IN ({', '.join([f"'{s}'" for s in subreddits_selected])})
    """
    posts = pd.read_sql(posts_query, engine)
    comments = pd.read_sql(comments_query, engine)
    return posts, comments

# Display posts in a scrollable table
def display_post_table(posts_data):
    st.write("### Filtered Dataset")
    st.dataframe(posts_data[['post_id', 'author', 'score', 'created_utc', 'subreddit', 'title']], height=200)

# Plot comments time series based on selected time interval
def plot_comment_timeseries(comments_data, selected_post_id, interval):
    comments_data['created_utc'] = pd.to_datetime(comments_data['created_utc'])
    comments_data = comments_data[comments_data['post_id'] == selected_post_id]
    comments_data.set_index('created_utc', inplace=True)
    comments_per_interval = comments_data.resample(interval).size().reset_index(name='comment_count')
    fig = px.bar(
        comments_per_interval,
        x='created_utc',
        y='comment_count',
        labels={"created_utc": "Time", "comment_count": "Number of Comments"},
        title=f"Number of comments over time - Post {selected_post_id}"
    )
    st.plotly_chart(fig)

# Calculate percentiles for comment times for each post with progress tracking
def calculate_comment_percentiles(posts_data, comments_data):
    percentiles = [5, 25, 50, 75]
    elapsed_data = []
    progress_bar = st.progress(0)
    total_posts = len(posts_data['post_id'].unique())
    for i, post_id in enumerate(posts_data['post_id'].unique()):
        post_time = posts_data.loc[posts_data['post_id'] == post_id, 'created_utc'].values[0]
        post_comments = comments_data[comments_data['post_id'] == post_id].copy()
        if not post_comments.empty:
            post_comments['elapsed'] = (post_comments['created_utc'] - post_time).dt.total_seconds()
            percentile_values = np.percentile(post_comments['elapsed'], percentiles)
            elapsed_row = {'post_id': post_id, 'score': posts_data.loc[posts_data['post_id'] == post_id, 'score'].values[0]}
            elapsed_row.update({f'{p}th_percentile_elapsed': v for p, v in zip(percentiles, percentile_values)})
            elapsed_data.append(elapsed_row)
        else:
            elapsed_row = {'post_id': post_id, 'score': posts_data.loc[posts_data['post_id'] == post_id, 'score'].values[0]}
            elapsed_row.update({f'{p}th_percentile_elapsed': None for p in percentiles})
            elapsed_data.append(elapsed_row)
        progress_bar.progress((i + 1) / total_posts)
    percentiles_df = pd.DataFrame(elapsed_data).dropna()
    for col in [f'{p}th_percentile_elapsed' for p in percentiles]:
        if col in percentiles_df.columns:
            percentiles_df[col] = percentiles_df[col] / percentiles_df[col].max()
    percentiles_df['score'] = percentiles_df['score'] / percentiles_df['score'].max()
    percentiles_df['subreddit'] = posts_data.set_index('post_id').loc[percentiles_df['post_id'], 'subreddit'].values
    unique_subreddits = posts_data['subreddit'].unique()
    for subreddit in unique_subreddits:
        percentiles_df[subreddit] = (percentiles_df['subreddit'] == subreddit).astype(int)
    percentiles_df.drop(columns=['subreddit'], inplace=True)
    percentiles_df.drop_duplicates(subset=['post_id'], inplace=True)
    progress_bar.empty()
    st.metric("Number of Posts", len(percentiles_df))
    return percentiles_df

# Define Autoencoder model in PyTorch
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=10):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, encoding_dim))
        self.decoder = nn.Sequential(nn.Linear(encoding_dim, 64), nn.ReLU(), nn.Linear(64, input_dim), nn.Sigmoid())
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Function to perform clustering with an autoencoder and K-Means
def perform_clustering(percentiles_df):
    feature_cols = [col for col in percentiles_df.columns if col != 'post_id']
    X = percentiles_df[feature_cols].values
    X = StandardScaler().fit_transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model = Autoencoder(X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        encoded, decoded = model(X_tensor)
        loss = criterion(decoded, X_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        compressed_features, _ = model(X_tensor)
        compressed_features = compressed_features.cpu().numpy()
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(compressed_features)
    percentiles_df['cluster'] = clusters
    clustered_df = percentiles_df[['post_id', 'score', 'cluster']]
    clustered_df.to_csv(Path(__file__).resolve().parent.parent / 'data' / 'unsupervised_post_clustering.csv', index=False)
    st.write("## Clustering complete and file saved to `./data`.")
    return clustered_df

# Streamlit dashboard setup
def main():
    st.set_page_config(layout="wide")
    st.title("Post Engagement Analysis Dashboard")

    # Sidebar filters
    st.sidebar.header("Select Filters")
    start_date = st.sidebar.date_input("Start Date", datetime(2024, 10, 15))
    end_date = st.sidebar.date_input("End Date", datetime(2024, 10, 31))
    engine = get_engine()
    all_subreddits = load_subreddits()
    subreddits_selected = st.sidebar.multiselect("Select Subreddits", options=all_subreddits, default=all_subreddits)
    posts_data, comments_data = load_data(engine, start_date, end_date, subreddits_selected)

    tabs = st.tabs(["Explore", "Cluster"])
    
    with tabs[0]:
        st.header("Totals")
        st.metric("Number of Posts", len(posts_data))
        st.metric("Number of Comments", len(comments_data))
        if not posts_data.empty:
            display_post_table(posts_data)
            selected_post_id = st.selectbox("Select a Post to View Comments", options=posts_data['post_id'].unique())
            interval_map = {
                '1 Second': '1S', '1 Minute': '1min', '5 Minutes': '5min', '1 Hour': '1h', '1 Day': '1D'
            }
            st.header(f"Comment Activity Over Time for Post: {selected_post_id}")
            st.write(f"Title: **{posts_data.loc[posts_data['post_id'] == selected_post_id, 'title'].values[0]}**")
            interval = st.selectbox("Select Time Interval", options=list(interval_map.keys()), index=1)
            interval_code = interval_map[interval]
            plot_comment_timeseries(comments_data, selected_post_id, interval_code)

    with tabs[1]:
        st.header("Cluster Posts")
        st.subheader("Building Database")
        percentiles_df = calculate_comment_percentiles(posts_data, comments_data)
        st.dataframe(percentiles_df)
        if st.button("Cluster Posts"):
            clustered_df = perform_clustering(percentiles_df)
            st.write("## Clustering Results")
            st.dataframe(clustered_df)

if __name__ == "__main__":
    main()
