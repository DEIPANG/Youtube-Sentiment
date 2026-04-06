import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re

# Download NLTK data (quietly)
nltk.download('vader_lexicon', quiet=True)

# 1. Setup Page UI
st.set_page_config(page_title="YouTube Sentiment App", layout="wide")
st.title("📊 YouTube Comments Sentiment Analyzer")
st.write("This app analyzes the sentiment of YouTube comments and assigns a Star Rating based on the emotion.")

# 2. Data Processing Function (Cached to run faster)
@st.cache_data
def load_and_process_data():
    # Load dataset
    df = pd.read_csv('YoutubeCommentsDataSet.csv')
    
    # Filter for English comments only
    def is_english(text):
        return bool(re.match(r'^[a-zA-Z0-9\s.,!?\'"]+$', str(text)))
    df = df[df['Comment'].apply(is_english)].dropna(subset=['Comment']).reset_index(drop=True)
    
    # VADER Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    compound_scores = []
    
    for text in df['Comment']:
        score = sia.polarity_scores(str(text))
        compound_scores.append(score['compound'])
        
    df['Compound_Score'] = compound_scores
    
    # Assign Star Rating (1 to 5 Stars)
    def get_star_rating(score):
        if score >= 0.5: return 5
        elif score >= 0.1: return 4
        elif score > -0.1: return 3
        elif score > -0.5: return 2
        else: return 1
        
    df['Star_Rating'] = df['Compound_Score'].apply(get_star_rating)
    return df

# Run the function
with st.spinner("Analyzing comments with AI..."):
    df = load_and_process_data()

# 3. Sidebar UI - Filter Function
st.sidebar.header("🔍 Filter Options")
selected_stars = st.sidebar.multiselect(
    "Select Star Rating to display:",
    options=[5, 4, 3, 2, 1],
    default=[5, 4, 3, 2, 1]
)

# Apply filter
filtered_df = df[df['Star_Rating'].isin(selected_stars)]

# 4. Main Layout & Visualizations
col1, col2 = st.columns([1, 2]) # Split screen into 2 columns

with col1:
    st.subheader("Chart: Star Rating Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x='Star_Rating', palette='viridis', ax=ax)
    ax.set_xlabel('Star Rating')
    ax.set_ylabel('Number of Comments')
    st.pyplot(fig)

with col2:
    st.subheader("Data: Filtered Comments")
    st.write(f"Showing {len(filtered_df)} comments.")
    # Show clean dataframe to users
    st.dataframe(filtered_df[['Comment', 'Sentiment', 'Star_Rating', 'Compound_Score']], use_container_width=True)
