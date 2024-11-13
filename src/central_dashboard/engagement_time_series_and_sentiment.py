#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# Adjust `PYTHONPATH` to include `src` for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.db_config import db_params
from config.color_palette import COLOR_PALETTE  # Import the centralized color palette

# Set up database connection
def get_engine():
    connection_string = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    return create_engine(connection_string)

# Query and load posts and comments
def load_data(engine, start_date, end_date):
    posts_query = f"""
    SELECT post_id, created_utc, subreddit, title, self_text FROM posts
    WHERE created_utc BETWEEN '{start_date}' AND '{end_date}'
    """
    comments_query = f"""
    SELECT comments.post_id, comments.created_utc, posts.subreddit, comments.body 
    FROM comments
    JOIN posts ON comments.post_id = posts.post_id
    WHERE comments.created_utc BETWEEN '{start_date}' AND '{end_date}'
    """
    posts = pd.read_sql(posts_query, engine)
    comments = pd.read_sql(comments_query, engine)

    # Combine title and self_text for posts into a single content column
    posts['content'] = posts['title'].fillna('') + " " + posts['self_text'].fillna('')
    posts = posts[['post_id', 'created_utc', 'subreddit', 'content']]

    # Rename comment body to content for consistency
    comments = comments.rename(columns={'body': 'content'})
    comments = comments[['post_id', 'created_utc', 'subreddit', 'content']]

    posts['created_utc'] = pd.to_datetime(posts['created_utc'])
    comments['created_utc'] = pd.to_datetime(comments['created_utc'])

    return posts, comments

# Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Function to calculate the average sentiment for posts based on all comments in a given time slot
def calculate_avg_sentiment_for_post_in_time_slot(time_slot, interval, posts_data, comments_data):
    try:
        end_time = time_slot + pd.to_timedelta(interval)
    except ValueError:
        from dateutil.relativedelta import relativedelta
        end_time = time_slot + relativedelta(months=int(interval[:-1])) if interval.endswith('M') else time_slot + pd.to_timedelta('1D')

    posts_in_time_slot = posts_data[(posts_data.index >= time_slot) & (posts_data.index < end_time)]['post_id'].unique()
    comments_in_time_slot = comments_data[comments_data['post_id'].isin(posts_in_time_slot)]

    sentiments = [sentiment_analyzer.polarity_scores(text)['compound'] for text in comments_in_time_slot['content']]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return avg_sentiment

# Function to calculate the average sentiment for comments in a given time slot
def calculate_avg_sentiment_for_time_slot(time_slot, interval, comments_data):
    try:
        end_time = time_slot + pd.to_timedelta(interval)
    except ValueError:
        from dateutil.relativedelta import relativedelta
        end_time = time_slot + relativedelta(months=int(interval[:-1])) if interval.endswith('M') else time_slot + pd.to_timedelta('1D')

    comments_in_time_slot = comments_data[(comments_data.index >= time_slot) & (comments_data.index < end_time)]
    sentiments = [sentiment_analyzer.polarity_scores(text)['compound'] for text in comments_in_time_slot['content']]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return avg_sentiment

# Plot posts per time slot using selected interval
def plot_posts_per_time_slot(posts_data, comments_data, subreddits_selected, interval, sentiment_analysis_active):
    if subreddits_selected:
        posts_filtered = posts_data[posts_data['subreddit'].isin(subreddits_selected)]
        comments_filtered = comments_data[comments_data['subreddit'].isin(subreddits_selected)]
    else:
        posts_filtered = posts_data.copy()
        comments_filtered = comments_data.copy()

    posts_filtered['created_utc'] = pd.to_datetime(posts_filtered['created_utc'])
    posts_filtered.set_index('created_utc', inplace=True)

    if sentiment_analysis_active:
        posts_per_time = posts_filtered.groupby(pd.Grouper(freq=interval)).size().reset_index(name='post_count')
        posts_per_time['avg_sentiment'] = posts_per_time['created_utc'].apply(
            lambda x: calculate_avg_sentiment_for_post_in_time_slot(x, interval, posts_filtered, comments_filtered)
        )

        fig = px.bar(
            posts_per_time,
            x='created_utc',
            y='post_count',
            color='avg_sentiment',
            color_continuous_scale=COLOR_PALETTE["continuous"],  # Use full plasma colormap
            labels={"created_utc": "Time Slot", "post_count": "Posts Count", "avg_sentiment": "Average Sentiment"},
            title=f"Posts per Time Slot ({interval}) Colored by Average Sentiment"
        )
    else:
        posts_per_time = posts_filtered.groupby([pd.Grouper(freq=interval), 'subreddit']).size().reset_index(name='post_count')
        fig = px.bar(
            posts_per_time,
            x='created_utc',
            y='post_count',
            color='subreddit',
            color_discrete_sequence=COLOR_PALETTE["categorical"],  # Use subreddit categorical colors
            labels={"created_utc": "Time Slot", "post_count": "Posts Count", "subreddit": "Subreddit"},
            title=f"Posts per Time Slot ({interval}) Colored by Subreddit"
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Post Count",
        plot_bgcolor=COLOR_PALETTE["background"],
        font=dict(color=COLOR_PALETTE["text"]),
        xaxis=dict(color=COLOR_PALETTE["axes"]),
        yaxis=dict(color=COLOR_PALETTE["axes"])
    )
    st.plotly_chart(fig)

# Plot comments per time slot using selected interval
def plot_comments_per_time_slot(comments_data, subreddits_selected, interval, sentiment_analysis_active):
    if subreddits_selected:
        comments_filtered = comments_data[comments_data['subreddit'].isin(subreddits_selected)]
    else:
        comments_filtered = comments_data.copy()

    comments_filtered['created_utc'] = pd.to_datetime(comments_filtered['created_utc'])
    comments_filtered.set_index('created_utc', inplace=True)

    if sentiment_analysis_active:
        comments_per_time = comments_filtered.groupby(pd.Grouper(freq=interval)).size().reset_index(name='comment_count')
        comments_per_time['avg_sentiment'] = comments_per_time['created_utc'].apply(
            lambda x: calculate_avg_sentiment_for_time_slot(x, interval, comments_filtered)
        )

        fig = px.bar(
            comments_per_time,
            x='created_utc',
            y='comment_count',
            color='avg_sentiment',
            color_continuous_scale=COLOR_PALETTE["continuous"],  # Use full plasma colormap
            labels={"created_utc": "Time Slot", "comment_count": "Comment Count", "avg_sentiment": "Average Sentiment"},
            title=f"Comments per Time Slot ({interval}) Colored by Average Sentiment"
        )
    else:
        comments_per_time = comments_filtered.groupby([pd.Grouper(freq=interval), 'subreddit']).size().reset_index(name='comment_count')
        fig = px.bar(
            comments_per_time,
            x='created_utc',
            y='comment_count',
            color='subreddit',
            color_discrete_sequence=COLOR_PALETTE["categorical"],  # Use subreddit categorical colors
            labels={"created_utc": "Time Slot", "comment_count": "Comment Count", "subreddit": "Subreddit"},
            title=f"Comments per Time Slot ({interval}) Colored by Subreddit"
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Comment Count",
        plot_bgcolor=COLOR_PALETTE["background"],
        font=dict(color=COLOR_PALETTE["text"]),
        xaxis=dict(color=COLOR_PALETTE["axes"]),
        yaxis=dict(color=COLOR_PALETTE["axes"])
    )
    st.plotly_chart(fig)

# Streamlit dashboard setup
def main():
    st.set_page_config(layout="wide")
    st.title("Time-Series Engagement and Sentiment Analysis")

    st.sidebar.header("Select Analysis Filters")
    start_date = st.sidebar.date_input("Start Date", datetime(2024, 10, 1))
    end_date = st.sidebar.date_input("End Date", datetime(2024, 11, 30))

    engine = get_engine()

    posts_data, comments_data = load_data(engine, start_date, end_date)

    all_subreddits = sorted(posts_data['subreddit'].unique())
    subreddits_selected = st.sidebar.multiselect("Select Subreddits", options=all_subreddits, default=all_subreddits)

    interval_map = {
        '5 Min': '5min', 
        'Hour': '1h',
        'Day': '1D', 
        'Week': '1W', 
        'Month': '1M'
    }
    interval_choice = st.sidebar.selectbox("Select Frequency Interval", list(interval_map.keys()))
    interval = interval_map[interval_choice]

    if 'sentiment_analysis_active' not in st.session_state:
        st.session_state.sentiment_analysis_active = False

    if st.sidebar.button("Begin Sentiment Analysis"):
        st.session_state.sentiment_analysis_active = True

    sentiment_analysis_active = st.session_state.sentiment_analysis_active

    st.subheader("Posts and Comments per Time Slot")
    plot_posts_per_time_slot(posts_data, comments_data, subreddits_selected, interval, sentiment_analysis_active)
    plot_comments_per_time_slot(comments_data, subreddits_selected, interval, sentiment_analysis_active)

    if sentiment_analysis_active:
        if st.sidebar.button("Reset Sentiment Analysis"):
            st.session_state.sentiment_analysis_active = False

if __name__ == "__main__":
    main()

