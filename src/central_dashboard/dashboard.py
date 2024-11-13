import sys
from pathlib import Path
import streamlit as st
from datetime import datetime

# Set up the PYTHONPATH to access other modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import your dashboard functions
from src.dashboards.flag_user_dashboard_copy import flag_user_dashboard
from src.dashboards.post_engagement_analysis_copy import post_engagement_analysis
from src.dashboards.swarm_detection_copy import swarm_detection
from src.dashboards.time_series_sentiment_dashboard_copy import time_series_sentiment_dashboard
from src.dashboards.user_clustering_and_community_detection_copy import user_clustering_and_community_detection
from src.dashboards.user_overlap_analysis_copy import user_overlap_analysis

# Define database and data filtering options here so they're accessible to all tabs
st.set_page_config(layout="wide", page_title="Reddit Interactions Dashboard")

st.sidebar.header("Global Filters")
start_date = st.sidebar.date_input("Start Date", datetime(2024, 10, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 10, 31))

# Load subreddits from JSON file
def load_subreddits():
    subreddits_path = Path(__file__).resolve().parent.parent / 'data' / 'target_subreddits.json'
    with open(subreddits_path) as f:
        subreddits = json.load(f).get("subreddits", [])
    return subreddits

subreddits_selected = st.sidebar.multiselect("Select Subreddits", options=load_subreddits(), default=load_subreddits())

# Define the main dashboard with tabs
def main():
    st.title("Centralized Dashboard")

    # Define tabs for each individual dashboard
    tab_names = ["Flag User Analysis", "Post Engagement", "Swarm Detection", "Time Series Sentiment", "User Clustering", "User Overlap"]
    tabs = st.tabs(tab_names)

    # Place each dashboard function in its respective tab
    with tabs[0]:
        st.header("Flag User Analysis")
        flag_user_dashboard(start_date, end_date, subreddits_selected)

    with tabs[1]:
        st.header("Post Engagement Analysis")
        post_engagement_analysis(start_date, end_date, subreddits_selected)

    with tabs[2]:
        st.header("Swarm Detection")
        swarm_detection(start_date, end_date, subreddits_selected)

    with tabs[3]:
        st.header("Time Series Sentiment Analysis")
        time_series_sentiment_dashboard(start_date, end_date, subreddits_selected)

    with tabs[4]:
        st.header("User Clustering and Community Detection")
        user_clustering_and_community_detection(start_date, end_date, subreddits_selected)

    with tabs[5]:
        st.header("User Overlap Analysis")
        user_overlap_analysis(start_date, end_date, subreddits_selected)

if __name__ == "__main__":
    main()
