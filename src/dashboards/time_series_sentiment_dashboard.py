#!/usr/bin/env python3

import sys
from pathlib import Path
import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# Adjust `PYTHONPATH` to include `src` for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import db_config

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
    posts = posts[['post_id', 'created_utc', 'subreddit', 'content']]  # Keep necessary columns

    # Rename comment body to content for consistency and keep subreddit
    comments = comments.rename(columns={'body': 'content'})
    comments = comments[['post_id', 'created_utc', 'subreddit', 'content']]

    # Concatenate posts and comments into a single DataFrame
    data = pd.concat([posts, comments], ignore_index=True)

    return data

# Initialize VADER sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Stacked bar plot for engagement by day and subreddit
def plot_engagement_by_day(data, subreddits_selected):
    # Filter data by selected subreddits
    if subreddits_selected:
        data = data[data['subreddit'].isin(subreddits_selected)]

    # Resample to daily counts per subreddit
    data['created_utc'] = pd.to_datetime(data['created_utc'])
    data.set_index('created_utc', inplace=True)
    daily_engagement = data.groupby([pd.Grouper(freq='D'), 'subreddit']).size().reset_index(name='engagement_count')

    # Plot stacked bar chart
    fig = px.bar(
        daily_engagement,
        x='created_utc',
        y='engagement_count',
        color='subreddit',
        labels={"created_utc": "Date", "engagement_count": "Engagement Count"},
        title="Daily Engagement by Subreddit"
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Engagement Count", barmode="stack")
    st.plotly_chart(fig)

# Calculate sentiment and group by interval with progress bar
def process_engagement_data(data, interval):
    data['created_utc'] = pd.to_datetime(data['created_utc'])
    data.set_index('created_utc', inplace=True)

    # Initial engagement data without sentiment
    engagement_data = data.resample(interval).agg(
        engagement=('content', 'count')  # Engagement is the count of posts + comments
    ).dropna()  # Drop intervals with no engagement

    engagement_data['avg_sentiment'] = 0  # Initialize with neutral sentiment

    # Process sentiment scores for each time slot in batches
    for index in tqdm(engagement_data.index, desc="Processing Sentiment by Interval"):
        interval_data = data[(data.index >= index) & (data.index < index + pd.Timedelta(interval))]

        if not interval_data.empty:
            sentiments = [sentiment_analyzer.polarity_scores(text)['compound'] for text in interval_data['content']]
            avg_sentiment = sum(sentiments) / len(sentiments)
        else:
            avg_sentiment = 0  # Neutral sentiment if no data

        engagement_data.at[index, 'avg_sentiment'] = avg_sentiment
        update_plot(engagement_data)

    return engagement_data

# Update the engagement plot dynamically in Streamlit
def update_plot(engagement_data):
    fig = px.bar(
        engagement_data,
        x=engagement_data.index,
        y="engagement",
        color="avg_sentiment",
        color_continuous_scale="RdYlGn",  # Red to green for negative to positive sentiment
        labels={"engagement": "Engagement (Post + Comment Count)", "avg_sentiment": "Avg Sentiment"},
        title="Dynamic Engagement and Sentiment Analysis"
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Engagement Count")
    placeholder.plotly_chart(fig)

# Streamlit dashboard setup
def main():
    st.title("Time-Series Engagement and Sentiment Analysis")

    # Sidebar filters
    st.sidebar.header("Select Analysis Filters")
    start_date = st.sidebar.date_input("Start Date", datetime(2024, 10, 1))
    end_date = st.sidebar.date_input("End Date", datetime(2024, 11, 30))

    # Database engine
    engine = get_engine()

    # Load data within selected date range
    data = load_data(engine, start_date, end_date)

    # Fetch distinct subreddits for filtering options
    all_subreddits = sorted(data['subreddit'].unique())
    subreddits_selected = st.sidebar.multiselect("Select Subreddits", options=all_subreddits, default=all_subreddits)

    # Display stacked bar plot of engagement by day
    st.subheader("Engagement by Day")
    plot_engagement_by_day(data, subreddits_selected)

    # Frequency interval options for sentiment analysis
    interval_map = {
        '5 Min': ('5', 'min'), 
        'Hour': ('1', 'H'), 
        'Day': ('1', 'D'), 
        'Week': ('1', 'W'), 
        'Month': ('1', 'M')
    }
    interval_choice = st.sidebar.selectbox("Select Frequency Interval", list(interval_map.keys()))
    interval_value, interval_unit = interval_map[interval_choice]
    interval = f"{interval_value}{interval_unit}"

    # Global placeholder for dynamic updates
    global placeholder  
    placeholder = st.empty()  # Create empty placeholder for sentiment plot

    # Add a "Go" button to start sentiment analysis after filtering
    if st.sidebar.button("Begin Sentiment Analysis"):
        # Process engagement data for sentiment analysis
        engagement_data = process_engagement_data(data, interval)

if __name__ == "__main__":
    main()
