# Reddit NLP Analysis Project

### Overview
This project is an end-to-end data science pipeline focused on performing Natural Language Processing (NLP) on Reddit data. It collects Reddit posts and comments using the PRAW library, stores the data in a PostgreSQL database, and uses Jupyter notebooks for data analysis, visualization, and NLP tasks. The project includes a custom Streamlit dashboard to visualize key metrics.

### Project Structure
- **Data Collection**: Uses the Reddit API via PRAW to retrieve and store data on posts and comments.
- **Database Management**: Stores data in PostgreSQL, creating relational tables optimized for efficient querying and analysis.
- **Data Analysis**: Uses Jupyter notebooks for exploratory data analysis (EDA), text preprocessing, and machine learning.
- **NLP Tasks**: Conducts NLP tasks such as sentiment analysis and topic modeling.
- **Visualization**: Creates interactive dashboards in Streamlit for exploring trends and insights.

### Prerequisites
- **Python 3.7+**
- **PostgreSQL**
- **Jupyter Notebook**
- **Git**
- **PRAW**: Install using `pip install praw`
- **NLTK, scikit-learn, and other NLP libraries**: Install with `pip install nltk scikit-learn`

### Repository Structure
```plaintext
reddit-nlp-analysis/
├── README.md
├── docs
│   └── swarm_detection.md
├── environment.yaml
└── src
    ├── __init__.py
    ├── config
    │   ├── __init__.py
    │   ├── color_palette.example.py
    │   ├── db_config.example.py
    │   ├── reddit_config.example.py
    ├── dashboards
    │   ├── __init__.py
    │   ├── flag_user_dashboard.py
    │   ├── post_engagement_analysis.py
    │   ├── swarm_detection.py
    │   ├── time_series_sentiment_dashboard.py
    │   ├── user_clustering_and_community_detection.py
    │   └── user_overlap_analysis.py
    ├── data
    │   ├── __init__.py
    │   ├── flagged_users.example.json
    │   ├── target_subreddits.example.json
    └── scrapers
        ├── __init__.py
        ├── fetch_flagged_user_data.py
        └── fetch_subreddit_data.py

