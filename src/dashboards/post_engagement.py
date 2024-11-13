import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

# Adjust `PYTHONPATH` to include `src` for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.db_config import db_params
from config.color_palette import COLOR_PALETTE  # Import custom color palette

# Set up database connection
def get_engine():
    connection_string = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    return create_engine(connection_string)

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
    st.write("## Available Posts")
    st.dataframe(posts_data[['post_id', 'author', 'score', 'created_utc', 'subreddit', 'title']], height=300)

# Plot comments time series based on selected time interval with color adjustments
def plot_comment_timeseries(comments_data, selected_post_id, interval):
    comments_data['created_utc'] = pd.to_datetime(comments_data['created_utc'])
    comments_data = comments_data[comments_data['post_id'] == selected_post_id]
    
    # Resample by chosen interval and count comments in each bin
    comments_data.set_index('created_utc', inplace=True)
    comments_per_interval = comments_data.resample(interval).size().reset_index(name='comment_count')
    
    # Plot
    fig = px.bar(
        comments_per_interval,
        x='created_utc',
        y='comment_count',
        labels={"created_utc": "Time", "comment_count": "Number of Comments"},
        title="Comments Over Time for Selected Post",
        color_discrete_sequence=[COLOR_PALETTE['primary']]
    )
    
    fig.update_layout(
        plot_bgcolor=COLOR_PALETTE['background'],
        paper_bgcolor=COLOR_PALETTE['background'],
        title_font_color=COLOR_PALETTE['text'],
        xaxis_title_font_color=COLOR_PALETTE['text'],
        yaxis_title_font_color=COLOR_PALETTE['text'],
        font_color=COLOR_PALETTE['text']
    )
    
    st.plotly_chart(fig)

# Streamlit dashboard setup
def main():
    st.set_page_config(layout="wide")
    st.title("Post and Comment Analysis Dashboard")

    # Sidebar filters
    st.sidebar.header("Select Filters")
    start_date = st.sidebar.date_input("Start Date", datetime(2024, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime(2024, 12, 31))

    # Set up database engine
    engine = get_engine()

    # Define subreddit options
    all_subreddits = ['politics', 'news', 'technology', 'worldnews', 'conspiracy']  # Example subreddits
    subreddits_selected = st.sidebar.multiselect("Select Subreddits", options=all_subreddits, default=all_subreddits)

    # Load data within the selected date range and subreddits
    posts_data, comments_data = load_data(engine, start_date, end_date, subreddits_selected)

    if posts_data.empty:
        st.error("No posts found for the selected filters!")
    else:
        # Display posts in a table and allow the user to select one
        display_post_table(posts_data)

        # Select a post
        selected_post_id = st.selectbox("Select a Post to View Comments", options=posts_data['post_id'].unique())
        
        # Select interval for time-series plot
        interval_map = {
            '1 Second': '1S',
            '1 Minute': '1min',
            '2 Minutes': '2min',
            '3 Minutes': '3min',
            '5 Minutes': '5min',
            '10 Minutes': '10min',
            '15 Minutes': '15min',
            '30 Minutes': '30min',
            '1 Hour': '1h',
            '2 Hours': '2h',
            '3 Hours': '3h',
            '6 Hours': '6h',
            '1 Day': '1D'
        }
        interval = st.selectbox("Select Time Interval", options=list(interval_map.keys()), index=5)
        interval_code = interval_map[interval]

        # Display comments time series
        st.write("## Comment Activity Over Time")
        plot_comment_timeseries(comments_data, selected_post_id, interval_code)

if __name__ == "__main__":
    main()
