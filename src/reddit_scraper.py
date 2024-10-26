import praw
import psycopg2
from psycopg2.extras import execute_values
import datetime
import time
import db_config
import reddit_config  # Import credentials from reddit_config.py

# Initialize Reddit using imported credentials
reddit = praw.Reddit(
    client_id=reddit_config.client_id,
    client_secret=reddit_config.client_secret,
    user_agent=reddit_config.user_agent
)

# List of subreddits to scrape
subreddits = ["news", "worldnews", "politics", "technology", "conspiracy", "inthenews"]

# PostgreSQL Database Configuration from db_config.py
conn = psycopg2.connect(**db_config.db_params)
cursor = conn.cursor()

# Database size limit in bytes (10 GB)
MAX_DB_SIZE = 10 * 1024 * 1024 * 1024

# Function to create tables if they don't exist
def create_tables():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        post_id TEXT PRIMARY KEY,
        subreddit TEXT,
        author TEXT,
        title TEXT,
        self_text TEXT,
        url TEXT,
        created_utc TIMESTAMP,
        score INTEGER,
        num_comments INTEGER,
        flair TEXT
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        comment_id TEXT PRIMARY KEY,
        post_id TEXT REFERENCES posts(post_id) ON DELETE CASCADE,
        parent_id TEXT,
        author TEXT,
        body TEXT,
        created_utc TIMESTAMP,
        score INTEGER
    )
    """)
    
    conn.commit()

# Function to check if database size exceeds the limit
def database_exceeds_size_limit():
    cursor.execute("SELECT pg_database_size(current_database())")
    db_size = cursor.fetchone()[0]
    return db_size > MAX_DB_SIZE

# Function to insert posts into the database
def insert_posts(posts):
    insert_query = """
    INSERT INTO posts (post_id, subreddit, author, title, self_text, url, created_utc, score, num_comments, flair)
    VALUES %s
    ON CONFLICT (post_id) DO NOTHING
    """
    execute_values(cursor, insert_query, posts)
    conn.commit()

# Function to insert comments into the database
def insert_comments(comments):
    insert_query = """
    INSERT INTO comments (comment_id, post_id, parent_id, author, body, created_utc, score)
    VALUES %s
    ON CONFLICT (comment_id) DO NOTHING
    """
    execute_values(cursor, insert_query, comments)
    conn.commit()

# Function to scrape posts and all comments from a subreddit
def scrape_subreddit(subreddit_name, limit=10):
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []
    comments_data = []

    for post in subreddit.hot(limit=limit):
        # Collect post data
        posts_data.append((
            post.id,
            subreddit_name,
            str(post.author),
            post.title,
            post.selftext,
            post.url,
            datetime.datetime.fromtimestamp(post.created_utc),
            post.score,
            post.num_comments,
            post.link_flair_text
        ))

        # Fetch all comments for each post
        post.comments.replace_more(limit=0)
        for comment in post.comments.list():
            comments_data.append((
                comment.id,
                post.id,
                comment.parent_id,
                str(comment.author),
                comment.body,
                datetime.datetime.fromtimestamp(comment.created_utc),
                comment.score
            ))

        # Insert data after each post to minimize memory usage
        insert_posts(posts_data)
        insert_comments(comments_data)
        posts_data.clear()
        comments_data.clear()

        # Pause between posts to keep ~98 requests per minute
        time.sleep(0.61)

# Main continuous data collection loop
def continuous_data_collection():
    create_tables()  # Ensure tables are created before data insertion

    try:
        while True:
            if database_exceeds_size_limit():
                print("Database size limit exceeded. Stopping data collection.")
                break

            for subreddit_name in subreddits:
                print(f"Scraping subreddit: {subreddit_name}")
                scrape_subreddit(subreddit_name, limit=1)  # Scraping one post at a time conservatively

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cursor.close()
        conn.close()
        print("Data collection stopped.")

# Run continuous data collection
continuous_data_collection()

