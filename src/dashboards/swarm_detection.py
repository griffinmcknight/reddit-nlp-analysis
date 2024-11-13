#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt  # Added import for matplotlib

# Adjust `PYTHONPATH` to include `src` for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import configuration modules, including the color palette
from config import db_config, reddit_config, color_palette  # Import color_palette
COLOR_PALETTE = color_palette.COLOR_PALETTE  # Reference the color palette dictionary

# Load the Plasma colormap from matplotlib
plasma_cmap = plt.cm.plasma(np.linspace(0, 1, 256))  # Define 256 colors along the Plasma colormap

# Utility function to convert RGBA to HEX for Streamlit/Plotly compatibility
def rgba_to_hex(rgba):
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))

# Update COLOR_PALETTE with continuous colormap
COLOR_PALETTE["continuous"] = [rgba_to_hex(color) for color in plasma_cmap]

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

# Function to establish database connection
def get_engine():
    try:
        connection_string = (
            f"postgresql+psycopg2://{db_config.db_params['user']}:{db_config.db_params['password']}@"
            f"{db_config.db_params['host']}:{db_config.db_params['port']}/{db_config.db_params['dbname']}"
        )
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None

# Function to load posts from the database
def load_posts(engine, start_date, end_date, subreddits):
    if not subreddits:
        st.warning("No subreddits selected.")
        return pd.DataFrame()
        
    subreddits_str = "', '".join(subreddits)
    posts_query = f"""
    SELECT post_id, created_utc, subreddit, score
    FROM posts
    WHERE created_utc BETWEEN '{start_date}' AND '{end_date}'
    AND subreddit IN ('{subreddits_str}')
    """
    try:
        posts = pd.read_sql(posts_query, engine)
        if posts.empty:
            st.warning("No posts found for the selected filters.")
            return pd.DataFrame()
        return posts[['post_id', 'created_utc', 'subreddit', 'score']]
    except Exception as e:
        st.error(f"Error loading posts data: {e}")
        return pd.DataFrame()

# Function to load comments from the database
def load_comments(engine, start_date, end_date, subreddits):
    if not subreddits:
        st.warning("No subreddits selected.")
        return pd.DataFrame()
        
    subreddits_str = "', '".join(subreddits)
    comments_query = f"""
    SELECT comments.comment_id, comments.post_id, comments.created_utc, posts.subreddit, comments.body, comments.score
    FROM comments
    JOIN posts ON comments.post_id = posts.post_id
    WHERE comments.created_utc BETWEEN '{start_date}' AND '{end_date}'
    AND posts.subreddit IN ('{subreddits_str}')
    """
    try:
        comments = pd.read_sql(comments_query, engine)
        if comments.empty:
            st.warning("No comments found for the selected filters.")
            return pd.DataFrame()
        comments = comments.rename(columns={'body': 'content'})
        return comments[['comment_id', 'post_id', 'created_utc', 'subreddit', 'score', 'content']]
    except Exception as e:
        st.error(f"Error loading comments data: {e}")
        return pd.DataFrame()

# Function to calculate bins and frequencies with a fixed number of bins
def calculate_bins_and_frequencies(data, score_column, num_bins=26, iqrm=1.5):
    # Calculate IQR and outlier thresholds
    q1 = np.percentile(data[score_column], 25)
    q3 = np.percentile(data[score_column], 75)
    iqr = q3 - q1
    lower_bound = q1 - iqrm * iqr
    upper_bound = q3 + iqrm * iqr

    # Separate main data and outliers
    main_data = data[(data[score_column] >= lower_bound) & (data[score_column] <= upper_bound)]
    lower_outliers = data[data[score_column] < lower_bound]
    upper_outliers = data[data[score_column] > upper_bound]

    # Calculate bin edges for main data
    main_min, main_max = main_data[score_column].min(), main_data[score_column].max()
    bin_edges = np.linspace(main_min, main_max, num_bins + 1)

    # Calculate frequencies for main data bins
    main_data_counts, _ = np.histogram(main_data[score_column], bins=bin_edges)

    # Frequency for lower and upper outliers
    lower_outliers_count = len(lower_outliers)
    upper_outliers_count = len(upper_outliers)

    return bin_edges, main_data_counts, lower_outliers_count, upper_outliers_count

# Function to plot histogram with colorized bars based on percentage
def plot_histogram_with_outliers(data, score_column, title, num_bins=26, iqrm=1.5):
    data = data.dropna(subset=[score_column])
    if data.empty:
        st.info(f"No data available to plot for {score_column}.")
        return

    # Calculate bins, frequencies, and outliers
    bin_edges, main_data_counts, lower_outliers_count, upper_outliers_count = calculate_bins_and_frequencies(data, score_column, num_bins, iqrm)

    # Total number of data points
    total_data_points = len(data)

    # Set a fixed, wide bar width to nearly fill the space
    visual_bin_width = 0.9 * (bin_edges[1] - bin_edges[0])  # A width that almost touches but leaves slight space

    # Create x-values for main bins as midpoints of each bin
    main_bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Calculate percentages for main bins
    main_bin_percentages = main_data_counts / total_data_points

    # Map percentages to colors
    colormap = COLOR_PALETTE["continuous"]
    num_colors = len(colormap)

    # Map percentages to color indices
    color_indices = (main_bin_percentages * (num_colors - 1)).astype(int)
    # Ensure indices are within valid range
    color_indices = np.clip(color_indices, 0, num_colors - 1)
    # Get colors for each bin
    bin_colors = [colormap[idx] for idx in color_indices]

    # Create the main data histogram as bars
    fig = go.Figure()

    # Plot main data bins with colors
    fig.add_trace(go.Bar(
        x=main_bin_centers,
        y=main_data_counts,
        name="Main Data",
        marker_color=bin_colors,  # Use colors per bin
        width=visual_bin_width
    ))

    # Handle outliers
    # For outliers, calculate percentage and map to color
    if lower_outliers_count > 0:
        lower_outlier_percentage = lower_outliers_count / total_data_points
        idx = int(lower_outlier_percentage * (num_colors - 1))
        idx = np.clip(idx, 0, num_colors - 1)
        lower_outlier_color = colormap[idx]

        fig.add_trace(go.Bar(
            x=[main_bin_centers[0] - iqrm * visual_bin_width],
            y=[lower_outliers_count],
            name=f"Lower Outliers {round(lower_outlier_percentage*100, 3)}%",
            marker_color=lower_outlier_color,
            width=visual_bin_width
        ))

    if upper_outliers_count > 0:
        upper_outlier_percentage = upper_outliers_count / total_data_points
        idx = int(upper_outlier_percentage * (num_colors - 1))
        idx = np.clip(idx, 0, num_colors - 1)
        upper_outlier_color = colormap[idx]

        fig.add_trace(go.Bar(
            x=[main_bin_centers[-1] + iqrm * visual_bin_width],
            y=[upper_outliers_count],
            name=f"Upper Outliers {round(upper_outlier_percentage*100, 3)}%",
            marker_color=upper_outlier_color,
            width=visual_bin_width
        ))

    # Layout adjustments for minimal gap
    fig.update_layout(
        title=title,
        xaxis_title=score_column,
        yaxis_title="Frequency",
        plot_bgcolor=COLOR_PALETTE["background"],
        font=dict(color=COLOR_PALETTE["axes"]),
        bargap=0.02  # Tiny gap for barely touching bars
    )
    st.plotly_chart(fig, use_container_width=True)

# Main Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("Reddit Engagement Dashboard")
    
    subreddits = load_target_subreddits()
    if not subreddits:
        st.stop()
    
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2024, 10, 20), min_value=datetime(2005, 6, 13), max_value=datetime.today())
    end_date = st.sidebar.date_input("End Date", value=datetime(2024, 11, 13), min_value=start_date, max_value=datetime.today())
    selected_subreddits = st.sidebar.multiselect("Select Subreddits", options=subreddits, default=subreddits)
    
    # User-adjustable bin count with default to 26
    num_bins = st.sidebar.number_input("Number of Bins", min_value=5, max_value=100, value=26, step=1)

    # User-defined IQR multiplier for highly skewed distributions
    iqr_multiplier = st.sidebar.number_input("Variable IQR multiplier for outlier detection", min_value=float(1), max_value=float(15), value=float(1.5), step=float(0.25))
    
    engine = get_engine()
    if engine is None:
        st.stop()
    
    start_date_str, end_date_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    with st.spinner("Loading data..."):
        posts = load_posts(engine, start_date_str, end_date_str, selected_subreddits)
        comments = load_comments(engine, start_date_str, end_date_str, selected_subreddits)
    
    if posts.empty and comments.empty:
        st.warning("No data found for the selected filters.")
        st.stop()
    
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    col1.metric("Number of Posts", len(posts))
    col2.metric("Number of Comments", len(comments))
    
    st.subheader("Histogram of Scores Received")
    st.markdown(f"<h4 style='color: {COLOR_PALETTE['primary']}'>Post Scores</h4>", unsafe_allow_html=True)
    plot_histogram_with_outliers(posts, 'score', "Distribution of Post Scores", num_bins, iqr_multiplier)

    st.markdown(f"<h4 style='color: {COLOR_PALETTE['primary']}'>Comment Scores</h4>", unsafe_allow_html=True)
    plot_histogram_with_outliers(comments, 'score', "Distribution of Comment Scores", num_bins, iqr_multiplier)

if __name__ == "__main__":
    main()
