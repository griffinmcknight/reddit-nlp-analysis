import streamlit as st
import pandas as pd
import plotly.express as px
import db_config  # Import the database configuration
from sqlalchemy import create_engine

# Database Connection using SQLAlchemy
def get_engine():
    params = db_config.db_params
    return create_engine(f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['dbname']}")

# Load Data from PostgreSQL
@st.cache_data
def load_data(query):
    engine = get_engine()
    data = pd.read_sql(query, engine)
    engine.dispose()
    return data

# Dashboard Layout
st.title("Reddit Data Exploratory Analysis Dashboard")
st.markdown("This dashboard provides an exploratory analysis of Reddit data.")

### 1. Post Volume Over Time
st.markdown("## 1. Post Volume Over Time")

# Query for post volume over time
time_granularity = st.selectbox("Select Time Granularity:", ["day", "week", "month"])
query_post_volume = f"""
SELECT DATE_TRUNC('{time_granularity}', created_utc) as period, 
       COUNT(*) as post_count
FROM posts
GROUP BY period
ORDER BY period;
"""
post_volume_data = load_data(query_post_volume)

# Plot post volume over time
fig1 = px.line(post_volume_data, x="period", y="post_count", title="Post Volume Over Time")
st.plotly_chart(fig1)

### 2. Top Authors and Engagement
st.markdown("## 2. Top Authors and Engagement")

# Query for top authors by post count and average score
query_top_authors = """
SELECT author, COUNT(*) as post_count, AVG(score) as avg_score, AVG(num_comments) as avg_comments
FROM posts
WHERE author IS NOT NULL
GROUP BY author
ORDER BY post_count DESC
LIMIT 10;
"""
top_authors_data = load_data(query_top_authors)

# Display table and plot for top authors
st.write("### Top 10 Authors by Post Count")
st.dataframe(top_authors_data)

fig2 = px.bar(top_authors_data, x="author", y="post_count", title="Top Authors by Post Count")
st.plotly_chart(fig2)

# Plot average engagement metrics for these authors
fig3 = px.bar(top_authors_data, x="author", y=["avg_score", "avg_comments"], 
              barmode="group", title="Top Authors: Average Score and Comments")
st.plotly_chart(fig3)

### 3. Subreddit Comparisons
st.markdown("## 3. Subreddit Comparisons")

# Query for subreddit comparison metrics
query_subreddit_comparison = """
SELECT subreddit, 
       COUNT(*) as post_count, 
       AVG(score) as avg_score, 
       AVG(num_comments) as avg_comments
FROM posts
GROUP BY subreddit
ORDER BY post_count DESC
LIMIT 10;
"""
subreddit_data = load_data(query_subreddit_comparison)

# Display subreddit comparison table
st.write("### Subreddit Comparison Metrics")
st.dataframe(subreddit_data)

# Plot subreddit comparisons
fig4 = px.bar(subreddit_data, x="subreddit", y="post_count", title="Post Frequency by Subreddit")
st.plotly_chart(fig4)

fig5 = px.bar(subreddit_data, x="subreddit", y=["avg_score", "avg_comments"], 
              barmode="group", title="Subreddit Engagement: Average Score and Comments")
st.plotly_chart(fig5)

st.markdown("### Insights")
st.markdown("This analysis highlights trends over time, top contributing authors, and differences across subreddits.")
