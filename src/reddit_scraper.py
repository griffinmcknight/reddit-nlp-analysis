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
subreddits = ["politics", "democrats", "news", "worldnews", "technology", "conspiracy", "inthenews", "conservative"]

# PostgreSQL Database Configuration from db_config.py
conn = psycopg2.connect(**db_config.db_params)
cursor = conn.cursor()

# Database size limit in bytes (10 GB)
MAX_DB_SIZE = 10 * 1024 * 1024 * 1024
BATCH_SIZE = 15  # Number of records to insert per batch

# Function to check if database size exceeds the limit
def database_exceeds_size_limit():
    cursor.execute("SELECT pg_database_size(current_database())")
    db_size = cursor.fetchone()[0]
    return db_size > MAX_DB_SIZE

# Function to insert posts into the database
def insert_post(post_data):
    insert_query = """
    INSERT INTO posts (post_id, subreddit, author, title, self_text, url, created_utc, score, num_comments, flair)
    VALUES %s
    ON CONFLICT (post_id) DO NOTHING
    """
    try:
        execute_values(cursor, insert_query, [post_data])
        conn.commit()
        print(f"Inserted post {post_data[0]} into the database.")
    except Exception as e:
        print(f"Error inserting post {post_data[0]}: {e}")

# Function to insert comments into the database
def insert_comments(comments):
    if not comments:
        return
    insert_query = """
    INSERT INTO comments (comment_id, post_id, parent_id, author, body, created_utc, score)
    VALUES %s
    ON CONFLICT (comment_id) DO NOTHING
    """
    try:
        execute_values(cursor, insert_query, comments)
        conn.commit()
        print(f"Inserted {len(comments)} comments into the database.")
    except Exception as e:
        print(f"Error inserting comments: {e}")

# Function to process a single post and its comments
def process_post(post, subreddit_name):
    # Prepare and insert post data immediately
    post_data = (
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
    )
    insert_post(post_data)

    # Fetch and process comments after the post is inserted
    comments_data = []
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

        # Insert comments in batches
        if len(comments_data) >= BATCH_SIZE:
            insert_comments(comments_data)
            comments_data.clear()

    # Insert any remaining comments
    insert_comments(comments_data)

    # Rate limiting to stay within Reddit API limits
    time.sleep(0.61)

# Function to scrape posts and all comments from a subreddit
def scrape_subreddit(subreddit_name, limit=1000):  # Increased limit for efficiency
    subreddit = reddit.subreddit(subreddit_name)

    # Scrape from `new` listing
    print(f"Scraping 'new' posts for {subreddit_name}")
    for post in subreddit.new(limit=limit):
        process_post(post, subreddit_name)

    # Scrape from `top` listing for specified timeframes
    for time_filter in ["day", "week", "month", "year"]:
        print(f"Scraping 'top' posts for {subreddit_name} - {time_filter}")
        for post in subreddit.top(time_filter=time_filter, limit=limit):
            process_post(post, subreddit_name)

# Main continuous data collection loop
def continuous_data_collection():
    try:
        while True:
            if database_exceeds_size_limit():
                print("Database size limit exceeded. Stopping data collection.")
                break

            for subreddit_name in subreddits:
                print(f"Scraping subreddit: {subreddit_name}")
                scrape_subreddit(subreddit_name, limit=1000)  # Scraping 100 posts at a time for efficiency

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cursor.close()
        conn.close()
        print("Data collection stopped.")

# Run continuous data collection
continuous_data_collection()

