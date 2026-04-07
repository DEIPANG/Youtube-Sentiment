import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
from wordcloud import WordCloud
from googleapiclient.discovery import build
from textblob import TextBlob

# --- 1. Page Config & Setup ---
st.set_page_config(page_title="YouTube Sentiment App", page_icon="🎥", layout="wide")
nltk.download('vader_lexicon', quiet=True)

def get_star_string(rating):
    return '⭐' * int(rating)

# --- 2. YouTube API Fetcher Function ---
def get_youtube_comments(video_url, api_key, max_results=100):
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

# --- 3. Data Processing Function ---
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
        
    df['Star_Rating_Num'] = df['Compound_Score'].apply(get_star_rating)
    df['Stars'] = df['Star_Rating_Num'].apply(get_star_string)
    df['Sentiment'] = df['Star_Rating_Num'].apply(lambda x: 'Positive' if x >=4 else ('Neutral' if x == 3 else 'Negative'))
    
    df['Token_Count'] = df['Comment'].apply(lambda x: len(TextBlob(str(x)).words))
    
    return df

# --- 4. Sidebar UI ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg", width=150)
st.sidebar.header("⚙️ App Settings")

data_source = st.sidebar.radio("Choose Data Source:", [
    "Use Demo CSV", 
    "Analyze Real YouTube Video",
    "Enter Custom Text"
])

df = pd.DataFrame() 

if data_source == "Analyze Real YouTube Video":
    st.sidebar.markdown("---")
    api_key = st.sidebar.text_input("Enter YouTube API Key:", type="password")
    video_url = st.sidebar.text_input("Enter YouTube Video URL:")
    
    if st.sidebar.button("Fetch & Analyze"):
        if not api_key or not video_url:
            st.sidebar.warning("Please provide both API Key and Video URL.")
        else:
            with st.spinner("Fetching comments from YouTube..."):
                raw_df = get_youtube_comments(video_url, api_key)
                if raw_df is not None:
                    df = process_sentiment(raw_df)
                    st.session_state['current_df'] = df
                    st.session_state['current_video_url'] = video_url 

elif data_source == "Enter Custom Text":
    st.sidebar.markdown("---")
    user_text = st.sidebar.text_area("Type or paste text here (one comment per line):", height=150)
    
    if st.sidebar.button("Analyze Text"):
        if not user_text.strip():
            st.sidebar.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing your text..."):
                lines = [line.strip() for line in user_text.split('\n') if line.strip()]
                raw_df = pd.DataFrame({'Comment': lines})
                df = process_sentiment(raw_df)
                st.session_state['current_df'] = df
                st.session_state['current_video_url'] = None 

else:
    if st.sidebar.button("Load Demo Data"):
        with st.spinner("Loading local CSV..."):
            raw_df = pd.read_csv('YoutubeCommentsDataSet.csv')
            df = process_sentiment(raw_df)
            st.session_state['current_df'] = df
            st.session_state['current_video_url'] = None 

if 'current_df' in st.session_state:
    df = st.session_state['current_df']

# --- 5. Main Dashboard UI ---
st.title("📊 YouTube Comments Sentiment Analyzer")
st.markdown("Discover the true emotions behind the comments! Fetch real-time data, use the demo, or enter your own text.")

if not df.empty:
    st.markdown("---")
    
    if st.session_state.get('current_video_url'):
        st.video(st.session_state['current_video_url'])
        st.markdown("---")

    avg_score = df['Star_Rating_Num'].mean()
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total English Comments", len(df))
    col_m2.metric("Average Star Rating", f"{avg_score:.1f} ⭐")
    col_m3.metric("Overall Sentiment", "Positive 😊" if avg_score >= 3.5 else ("Neutral 😐" if avg_score >= 2.5 else "Negative 😠"))

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⭐ Star Rating Distribution")
        fig_stars, ax_stars = plt.subplots(figsize=(8, 5))
        order = ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐', '⭐']
        sns.countplot(data=df, y='Stars', order=order, palette='viridis', ax=ax_stars)
        ax_stars.set_xlabel("Number of Comments")
        ax_stars.set_ylabel("")
        st.pyplot(fig_stars)
        
    with col2:
        st.subheader("☁️ Word Cloud (Common Themes)")
        all_words = ' '.join(df['Comment'].astype(str).tolist())
        try:
            wordcloud = WordCloud(width=800, height=500, background_color='white', colormap='Set2').generate(all_words)
            fig_wc, ax_wc = plt.subplots(figsize=(8, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
        except ValueError:
            st.info("Not enough words to generate a word cloud.")

    st.markdown("---")
    
    st.subheader("🔍 Explore the Data")
    
    star_options = ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐', '⭐']
    selected_stars = st.multiselect("Filter by Star Rating:", options=star_options, default=star_options)
    
    filtered_df = df[df['Stars'].isin(selected_stars)]
    
    st.write(f"Showing **{len(filtered_df)}** comments based on your filter:")
    st.dataframe(filtered_df[['Stars', 'Sentiment', 'Token_Count', 'Comment', 'Compound_Score']], use_container_width=True)
    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name='analyzed_results.csv',
        mime='text/csv',
    )

    # --- 6. NLP Deep Dive Inspector ---
    st.markdown("---")
    st.subheader("🪄 NLP Deep Dive Inspector")
    st.markdown("Select a comment from the filtered list above to analyze its grammar and structure.")
    
    if not filtered_df.empty:
        comment_list = filtered_df['Comment'].tolist()
        selected_comment = st.selectbox("Select a comment to inspect:", comment_list)
        
        if selected_comment:
            inspector_blob = TextBlob(selected_comment)
            
            col_nlp1, col_nlp2 = st.columns(2)
            
            with col_nlp1:
                st.info("✨ Spell Check Suggestion")
                corrected_text = inspector_blob.correct()
                if str(corrected_text) == selected_comment:
                    st.success("The spelling looks perfect! ✅")
                else:
                    st.warning(f"**Did you mean:** {corrected_text}")
                    
            with col_nlp2:
                st.info("🏷️ Parts of Speech (POS) Tags")
                with st.expander("View all tags for this sentence"):
                    for word, tag in inspector_blob.tags:
                        st.markdown(f"- **{word}** : `{tag}`")
else:
    st.info("👈 Please use the Sidebar to load data, fetch a video, or enter custom text!")
