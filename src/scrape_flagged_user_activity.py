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

# Load flagged users from file
def load_flagged_users():
    with open("flagged_users.txt", "r") as file:
        return [line.strip() for line in file if line.strip()]

# Connect to PostgreSQL database
conn = psycopg2.connect(**db_config.db_params)
cursor = conn.cursor()

# Define table creation for flagged user interactions if it doesn't exist
def create_flagged_user_interactions_table():
    create_table_query = """
    CREATE TABLE IF NOT EXISTS flagged_user_interactions (
        id SERIAL PRIMARY KEY,
        username TEXT,
        interaction_type TEXT,
        subreddit TEXT,
        created_utc TIMESTAMPTZ,
        content TEXT,
        post_id TEXT,
        parent_id TEXT,
        score INTEGER
    );
    """
    cursor.execute(create_table_query)
    conn.commit()

# Function to insert interactions into the database
def insert_interactions(interactions):
    insert_query = """
    INSERT INTO flagged_user_interactions (username, interaction_type, subreddit, created_utc, content, post_id, parent_id, score)
    VALUES %s
    ON CONFLICT DO NOTHING;
    """
    try:
        execute_values(cursor, insert_query, interactions)
        conn.commit()
        print(f"Inserted {len(interactions)} interactions into the database.")
    except Exception as e:
        print(f"Error inserting interactions: {e}")

# Scrape posts and comments for each flagged user
def scrape_user_interactions(usernames, batch_size=10):
    all_interactions = []
    for username in usernames:
        try:
            reddit_user = reddit.redditor(username)
            print(f"Scraping interactions for user: {username}")

            # Fetch posts
            for submission in reddit_user.submissions.new(limit=None):
                all_interactions.append((
                    username,
                    'post',
                    submission.subreddit.display_name,
                    datetime.datetime.fromtimestamp(submission.created_utc),
                    submission.title + " " + submission.selftext,
                    submission.id,
                    None,  # Parent ID is None for posts
                    submission.score
                ))

            # Fetch comments
            for comment in reddit_user.comments.new(limit=None):
                all_interactions.append((
                    username,
                    'comment',
                    comment.subreddit.display_name,
                    datetime.datetime.fromtimestamp(comment.created_utc),
                    comment.body,
                    comment.link_id,
                    comment.parent_id,
                    comment.score
                ))

            # Insert interactions in batches to avoid large transactions
            if len(all_interactions) >= batch_size:
                insert_interactions(all_interactions)
                all_interactions.clear()
                
            # Respect Reddit’s rate limit
            time.sleep(1)

        except Exception as e:
            print(f"Error scraping user {username}: {e}")
            time.sleep(2)  # Small delay in case of an error

    # Insert any remaining interactions
    if all_interactions:
        insert_interactions(all_interactions)

# Main function to control the workflow
def main():
    # Create table if it does not exist
    create_flagged_user_interactions_table()

    # Load flagged users
    flagged_users = load_flagged_users()
    print(f"Loaded {len(flagged_users)} flagged users.")

    # Scrape and insert interactions for each flagged user
    scrape_user_interactions(flagged_users)

# Run the script
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cursor.close()
        conn.close()
        print("Database connection closed.")

