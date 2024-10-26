# Reddit NLP Analysis Project

### Overview
This project is an end-to-end data science pipeline focused on performing Natural Language Processing (NLP) on Reddit data. It collects Reddit posts and comments using the PRAW library, stores the data in a PostgreSQL database, and uses Jupyter notebooks for data analysis, visualization, and NLP tasks. The project includes a Tableau dashboard to visualize key metrics, as well as interaction with a local Large Language Model (LLM) for additional analysis.

### Project Structure
- **Data Collection**: Uses the Reddit API via PRAW to retrieve and store data on posts and comments.
- **Database Management**: Stores data in PostgreSQL, creating relational tables optimized for efficient querying and analysis.
- **Data Analysis**: Uses Jupyter notebooks for exploratory data analysis (EDA), text preprocessing, and machine learning.
- **NLP Tasks**: Conducts NLP tasks such as sentiment analysis and topic modeling.
- **Visualization**: Creates interactive dashboards in Tableau for exploring trends and insights.
- **LLM Integration**: Integrates a local LLM (e.g., Llama) for personalized insights based on classified Reddit data.

### Prerequisites
- **Python 3.7+**
- **PostgreSQL**
- **Jupyter Notebook**
- **Tableau** (optional, for visualization)
- **Git**
- **PRAW**: Install using `pip install praw`
- **NLTK, scikit-learn, and other NLP libraries**: Install with `pip install nltk scikit-learn`

### Repository Structure
```plaintext
reddit-nlp-analysis/
├── notebooks/              # Jupyter notebooks for EDA and analysis
├── src/                    # Python scripts for data collection and processing
├── README.md               # Project overview and setup instructions
├── .gitignore              # Files and directories to be ignored by Git
└── requirements.txt        # List of Python dependencies

