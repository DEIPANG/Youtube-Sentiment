import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
from googleapiclient.discovery import build

# --- Setup & Configuration ---
st.set_page_config(page_title="YouTube Sentiment Analyzer", page_icon="🎥", layout="wide")
nltk.download('vader_lexicon', quiet=True)

# --- Custom CSS for Styling & Hover Effects ---
st.markdown("""
<style>
    /* Header styling */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #333;
        margin-bottom: 30px;
    }
    .header-logo {
        font-size: 24px;
        font-weight: bold;
        color: #FF0000;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-links a {
        color: #E0E0E0;
        text-decoration: none;
        margin-left: 20px;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    .header-links a:hover {
        color: #FF0000;
    }
    
    /* Video Card Styling */
    .video-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid #333;
        cursor: pointer;
        min-height: 200px;
    }
    .video-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(255, 0, 0, 0.2);
        border-color: #FF0000;
    }
    .video-card img {
        width: 100%;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .video-card h4 {
        font-size: 16px;
        color: #FFF;
        margin-bottom: 5px;
    }
    .video-card p {
        font-size: 12px;
        color: #AAA;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def get_star_string(rating):
    return '⭐' * int(rating)

def get_youtube_comments(video_url, api_key, max_results=50):
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
    if not video_id_match:
        return None
    video_id = video_id_match.group(1)
    
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText"
        )
        response = request.execute()
        
        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            
        return pd.DataFrame({'Comment': comments})
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def process_sentiment(df):
    def is_english(text):
        return bool(re.match(r'^[a-zA-Z0-9\s.,!?\'"]+$', str(text)))
    
    df = df[df['Comment'].apply(is_english)].dropna(subset=['Comment']).reset_index(drop=True)
    
    if df.empty:
        return df
        
    sia = SentimentIntensityAnalyzer()
    compound_scores = []
    
    for text in df['Comment']:
        score = sia.polarity_scores(str(text))
        compound_scores.append(score['compound'])
        
    df['Compound_Score'] = compound_scores
    
    def get_star_rating(score):
        if score >= 0.5: return 5
        elif score >= 0.1: return 4
        elif score > -0.1: return 3
        elif score > -0.5: return 2
        else: return 1
        
    df['Stars_Num'] = df['Compound_Score'].apply(get_star_rating)
    df['Stars'] = df['Stars_Num'].apply(get_star_string)
    df['Sentiment'] = df['Stars_Num'].apply(lambda x: 'Positive' if x >=4 else ('Neutral' if x == 3 else 'Negative'))
    
    return df

# --- Render Custom Header ---
st.markdown("""
<div class="header-container">
    <div class="header-logo">
        ▶ YouTube NLP Analyzer
    </div>
    <div class="header-links">
        <a href="#">Dashboard</a>
        <a href="#">My Projects</a>
        <a href="#">Settings</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main Input Area ---
col_space1, col_input, col_space2 = st.columns([1, 2, 1])
with col_input:
    api_key = st.text_input("YouTube API Key (Required for live data)", type="password", placeholder="Enter your Google Cloud API Key...")
    video_url = st.text_input("Paste YouTube Video Link Here", placeholder="https://www.youtube.com/watch?v=...")

st.markdown("<br>", unsafe_allow_html=True)

# --- View State Logic ---
if not video_url:
    # State 1: Recommended Videos Grid (Initial State)
    st.subheader("Recommended for Analysis")
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    
    def render_video_card(col, title, desc):
        col.markdown(f"""
        <div class="video-card">
            <div style="height: 100px; background-color: #333; border-radius: 5px; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; color: #666;">Thumbnail</div>
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    render_video_card(c1, "Tech Review 2024", "Analyze gadget opinions")
    render_video_card(c2, "Gaming Highlights", "Track audience reactions")
    render_video_card(c3, "Music Video Launch", "Fan sentiment breakdown")
    render_video_card(c4, "Movie Trailer", "Hype vs Disappointment")

else:
    # State 2: Analysis Results
    if not api_key:
        st.warning("Please enter your API key to fetch live comments.")
    else:
        with st.spinner("Fetching and analyzing comments..."):
            raw_df = get_youtube_comments(video_url, api_key)
            
            if raw_df is not None and not raw_df.empty:
                df = process_sentiment(raw_df)
                
                # --- Embedded Video ---
                st.video(video_url)
                st.markdown("---")
                
                # --- Star Rating Filter ---
                st.subheader("Filter by Star Rating")
                star_options = ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐', '⭐']
                selected_stars = st.multiselect("Select ratings to display:", options=star_options, default=star_options)
                
                filtered_df = df[df['Stars'].isin(selected_stars)]
                
                # --- Data Table ---
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("Comment Data")
                st.dataframe(
                    filtered_df[['Stars', 'Sentiment', 'Comment', 'Compound_Score']], 
                    use_container_width=True,
                    height=300
                )
                
                # --- Sentiment Bar Chart ---
                st.markdown("---")
                st.subheader("Sentiment Distribution")
                
                sentiment_counts = df['Sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#0E1117') 
                ax.set_facecolor('#0E1117')
                
                colors = ['#00C853', '#9E9E9E', '#FF3D00'] # Green, Gray, Red
                
                bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors, width=0.5)
                
                ax.set_ylabel('Number of Comments', color='white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                
                for spine in ax.spines.values():
                    spine.set_edgecolor('#333')
                
                st.pyplot(fig)
                
            else:
                st.error("Could not fetch comments or the video has no comments.")
