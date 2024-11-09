import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import networkx as nx
from pyvis.network import Network
from db_config import db_params  # Ensure db_config.py has correct db connection details

# Database Connection
def get_connection():
    conn = psycopg2.connect(**db_params)
    return conn

# Load Data from PostgreSQL
@st.cache_data
def load_data(query):
    conn = get_connection()
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# Dashboard Layout
st.title("Reddit Data Dashboard")
st.markdown("### Overview of Scraped Reddit Data")

# Display Subreddit Distribution
query_subreddit = """
SELECT subreddit, COUNT(*) as post_count
FROM posts
GROUP BY subreddit
ORDER BY post_count DESC;
"""
subreddit_data = load_data(query_subreddit)
fig1 = px.bar(subreddit_data, x="subreddit", y="post_count", title="Posts by Subreddit")
st.plotly_chart(fig1)

# Display Top Posts
query_top_posts = """
SELECT title, subreddit, score, num_comments
FROM posts
ORDER BY score DESC
LIMIT 10;
"""
top_posts_data = load_data(query_top_posts)
st.write("### Top 10 Posts by Score")
st.dataframe(top_posts_data)

# Comments Analysis
query_comments = """
SELECT p.subreddit, COUNT(c.comment_id) as comment_count
FROM comments c
JOIN posts p ON c.post_id = p.post_id
GROUP BY p.subreddit
ORDER BY comment_count DESC;
"""
comments_data = load_data(query_comments)
fig2 = px.pie(comments_data, names="subreddit", values="comment_count", title="Comments Distribution by Subreddit")
st.plotly_chart(fig2)

# Posts Over Time (Adjust based on your time range)
query_time_series = """
SELECT DATE(created_utc) as date, COUNT(*) as post_count
FROM posts
GROUP BY date
ORDER BY date;
"""
time_series_data = load_data(query_time_series)
fig3 = px.line(time_series_data, x="date", y="post_count", title="Posts Over Time")
st.plotly_chart(fig3)

# Network Analysis Section
st.markdown("### Network Analysis of User Interactions")

# Query to find user interactions across posts and subreddits
query_user_interactions = """
SELECT p.subreddit, p.post_id, c.author as commenter, p.author as post_author
FROM comments c
JOIN posts p ON c.post_id = p.post_id
WHERE c.author IS NOT NULL AND p.author IS NOT NULL
"""
user_interactions = load_data(query_user_interactions)

# Create a network graph with users as nodes and interactions as edges
G = nx.Graph()

# Add nodes and edges based on interactions
for _, row in user_interactions.iterrows():
    commenter = row['commenter']
    post_author = row['post_author']
    if commenter != post_author:
        # Edge between commenter and post author if they interact on the same post
        G.add_edge(commenter, post_author, subreddit=row['subreddit'])

# Visualize using Pyvis in Streamlit
net = Network(notebook=True, height="500px", width="100%", bgcolor="#222222", font_color="white")
net.from_nx(G)

# Save and display the graph
net.show("user_network.html")
st.write("### User Interaction Network")
st.markdown("This network shows interactions among users based on shared posts or comments.")
st.markdown("Nodes represent users, and edges represent shared posts or comments in the same subreddit.")
st.markdown("View the network to analyze interconnected user clusters and repeated interactions.")

# Display the interactive network graph in Streamlit
with open("user_network.html", "r", encoding="utf-8") as f:
    components.html(f.read(), height=600)

# Temporal Analysis of Active Users
query_user_activity = """
SELECT author, DATE(created_utc) as date, COUNT(*) as activity_count
FROM (
    SELECT author, created_utc FROM posts
    UNION ALL
    SELECT author, created_utc FROM comments
) as user_activity
WHERE author IS NOT NULL
GROUP BY author, date
ORDER BY activity_count DESC
LIMIT 10;
"""
user_activity_data = load_data(query_user_activity)

fig4 = px.line(user_activity_data, x="date", y="activity_count", color="author",
               title="User Activity Over Time for Top Active Users")
st.plotly_chart(fig4)

st.write("### Data Insights")
st.markdown("Explore patterns and trends based on subreddit posts and comments.")
