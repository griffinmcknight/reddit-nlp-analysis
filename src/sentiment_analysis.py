import streamlit as st
import pandas as pd
import plotly.express as px
import db_config
import subprocess
from sqlalchemy import create_engine
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

# Function to generate a user-friendly description for topics using Ollama Llama
def generate_topic_description(keywords):
    prompt = f"Provide a concise, user-friendly topic description based on the following keywords: {', '.join(keywords)}."
    result = subprocess.run(
        ["ollama", "generate", "llama2", "-p", prompt],
        capture_output=True,
        text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "Description generation failed."

# Database Connection using SQLAlchemy
def get_engine():
    params = db_config.db_params
    return create_engine(f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}:{params['dbname']}")

# Load Data from PostgreSQL
@st.cache_data
def load_data(query):
    engine = get_engine()
    data = pd.read_sql(query, engine)
    engine.dispose()
    return data

# Dashboard Layout
st.title("Reddit Topic Modeling Dashboard with Ollama-Generated Descriptions")
st.markdown("This dashboard provides topic modeling analysis for Reddit posts and leverages Ollama to generate user-friendly topic descriptions.")

### Load and Preprocess Data
query_posts = "SELECT post_id, title, self_text, created_utc, subreddit FROM posts LIMIT 5000;"
posts_data = load_data(query_posts)

# Combine title and self_text for modeling
posts_data['full_text'] = posts_data['title'] + ' ' + posts_data['self_text']
posts_data['created_utc'] = pd.to_datetime(posts_data['created_utc'])
texts = posts_data['full_text'].fillna("").tolist()

### 1. Latent Dirichlet Allocation (LDA) Topic Modeling with Ollama Generalization
st.markdown("## 1. Latent Dirichlet Allocation (LDA) with Ollama Generalization")

# Preprocess text using CountVectorizer
count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
doc_term_matrix = count_vectorizer.fit_transform(texts)

# LDA Model
num_topics = st.slider("Select Number of Topics for LDA", 2, 20, 5)
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(doc_term_matrix)

# Display LDA Topics and Generate Ollama Descriptions
topic_descriptions = {}
for topic_idx, topic in enumerate(lda_model.components_):
    keywords = [count_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
    description = generate_topic_description(keywords)
    topic_descriptions[f"Topic {topic_idx}"] = {
        "keywords": keywords,
        "description": description
    }

st.write("### LDA Topics with User-Friendly Descriptions")
st.write(pd.DataFrame(topic_descriptions).T)

### 2. Subreddit-Specific Topics using LDA with Ollama Generalization
st.markdown("## 2. Subreddit-Specific Topics with Ollama Generalization")

subreddits = posts_data['subreddit'].unique()
selected_subreddit = st.selectbox("Choose a subreddit for topic modeling:", subreddits)

# Filter data for the selected subreddit
subreddit_data = posts_data[posts_data['subreddit'] == selected_subreddit]
subreddit_texts = subreddit_data['full_text'].fillna("").tolist()

# Vectorize and apply LDA for the specific subreddit
subreddit_doc_term_matrix = count_vectorizer.fit_transform(subreddit_texts)
subreddit_lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
subreddit_lda_model.fit(subreddit_doc_term_matrix)

# Display subreddit-specific topics with Ollama-generated descriptions
subreddit_topic_descriptions = {}
for topic_idx, topic in enumerate(subreddit_lda_model.components_):
    keywords = [count_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
    description = generate_topic_description(keywords)
    subreddit_topic_descriptions[f"Topic {topic_idx}"] = {
        "keywords": keywords,
        "description": description
    }

st.write(f"### Topics in '{selected_subreddit}' Subreddit with User-Friendly Descriptions")
st.write(pd.DataFrame(subreddit_topic_descriptions).T)

### 3. Trending Topics Over Time with BERTopic and Ollama Generalization
st.markdown("## 3. Trending Topics Over Time with BERTopic and Ollama Generalization")

# Prepare data for trend analysis
posts_data['year_month'] = posts_data['created_utc'].dt.to_period('M').astype(str)

# Apply BERTopic for advanced topic modeling with trend tracking
@st.cache_data
def apply_bertopic(texts):
    topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=False)
    topics, _ = topic_model.fit_transform(texts)
    return topic_model, topics

topic_model, topics = apply_bertopic(texts)
posts_data['topic'] = topics

# Generate descriptions for BERTopic topics
topic_info = topic_model.get_topic_info()
for topic_idx in topic_info["Topic"].unique():
    if topic_idx != -1:
        keywords = topic_model.get_topic(topic_idx)[:10]
        keywords = [word for word, _ in keywords]
        description = generate_topic_description(keywords)
        topic_info.loc[topic_info["Topic"] == topic_idx, "Description"] = description

st.write("### Trending Topics with User-Friendly Descriptions")
st.write(topic_info)

# Group by topic and month
topic_trends = posts_data.groupby(['year_month', 'topic']).size().reset_index(name='count')

# Plot topics over time
fig = px.line(topic_trends, x='year_month', y='count', color='topic', title="Trending Topics Over Time")
st.plotly_chart(fig)

# Display insights section
st.markdown("### Insights")
st.markdown("""
This topic modeling analysis uses Latent Dirichlet Allocation (LDA) and BERTopic to uncover key topics discussed on Reddit.
* **LDA** provides broad topics based on the entire dataset and specific subreddits.
* **BERTopic** enables trend tracking for topic frequencies over time, helping to identify emerging discussions.
* **Ollama-powered descriptions** turn raw keywords into user-friendly topic summaries, making topics easier to understand.
""")
