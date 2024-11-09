import psycopg2
import pandas as pd
import db_config

def connect_db():
    try:
        conn = psycopg2.connect(**db_config.db_params)
        print("Database connection successful")
        return conn
    except Exception as e:
        print("Database connection failed:", e)

def fetch_data(query):
    conn = connect_db()
    try:
        # Read query result into a DataFrame
        df = pd.read_sql_query(query, conn)
        print(f"Fetched {len(df)} records")
    except Exception as e:
        print("Failed to fetch data:", e)
        df = pd.DataFrame()  # Return an empty DataFrame on failure
    finally:
        conn.close()  # Ensure connection is closed after the query
    return df