import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from test import analyze_sentiment, sia

# Set page config
st.set_page_config(
    page_title="Instagram Mood Analyzer Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Instagram Mood Analyzer Dashboard")
st.markdown("""
This dashboard shows your Instagram activity mood analysis and emotional patterns over time.
""")

# Load and process data
@st.cache_data
def load_data():
    folder_path = "data"
    all_data = []
    
    # Load all JSON files and process them
    json_files = {
        "likes": "insta_mood_mock_likes.json",
        "reels": "insta_mood_mock_reels.json",
        "messages": "insta_mood_mock_messages.json",
        "ads": "insta_mood_mock_ads_and_interests.json",
        "videos": "insta_mood_mock_watch_history.json"
    }
    
    for source, filename in json_files.items():
        try:
            with open(os.path.join(folder_path, filename)) as f:
                data = json.load(f)
                if source == "messages":
                    for convo in data:
                        if 'messages' in convo:
                            for msg in convo['messages']:
                                msg['source'] = source
                                msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
                                msg['date'] = msg['timestamp'].date()
                                s = analyze_sentiment(msg.get('content', ''))
                                msg['mood'] = s['label']
                                msg['mood_score'] = s['compound']
                                all_data.append(msg)
                else:
                    for item in data:
                        item['source'] = source
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        item['date'] = item['timestamp'].date()
                        if source != "videos":
                            s = analyze_sentiment(item.get('caption', '') or item.get('content', ''))
                            item['mood'] = s['label']
                            item['mood_score'] = s['compound']
                        else:
                            item['mood'] = "neutral"
                            item['mood_score'] = 0.0
                        all_data.append(item)
        except Exception as e:
            st.error(f"Error loading {filename}: {str(e)}")
    
    return pd.DataFrame(all_data)

# Load the data
df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['date'].min(), df['date'].max()),
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

sources = st.sidebar.multiselect(
    "Select Sources",
    options=df['source'].unique(),
    default=df['source'].unique()
)

# Filter data based on sidebar selections
mask = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1]) & (df['source'].isin(sources))
filtered_df = df[mask]

# Create two columns for the layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Mood Trends Over Time")
    
    # Calculate daily mood scores
    daily_mood = filtered_df.groupby('date')['mood_score'].mean().reset_index()
    daily_mood['mood_change'] = daily_mood['mood_score'].diff()
    daily_mood['anomaly'] = daily_mood['mood_change'].apply(lambda x: True if x is not None and x < -0.5 else False)
    
    # Create mood trend line chart
    fig = px.line(daily_mood, x='date', y='mood_score',
                  title='Daily Average Mood Score',
                  labels={'mood_score': 'Mood Score', 'date': 'Date'})
    
    # Add anomaly points
    anomaly_df = daily_mood[daily_mood['anomaly']]
    fig.add_scatter(x=anomaly_df['date'], y=anomaly_df['mood_score'],
                   mode='markers',
                   marker=dict(color='red', size=10),
                   name='Mood Anomalies')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display anomaly dates
    if not anomaly_df.empty:
        st.warning("🚨 Mood Anomalies Detected")
        for _, row in anomaly_df.iterrows():
            st.write(f"- {row['date']}: Significant mood drop detected")

with col2:
    st.subheader("🎭 Mood Distribution by Source")
    
    # Create mood distribution pie chart
    mood_dist = filtered_df.groupby(['source', 'mood']).size().unstack(fill_value=0)
    fig = px.pie(values=mood_dist.values.flatten(),
                 names=mood_dist.columns.repeat(len(mood_dist)),
                 title='Mood Distribution Across Sources')
    st.plotly_chart(fig, use_container_width=True)

# Mood Summary Table
st.subheader("📊 Mood Summary Table")
summary = filtered_df.groupby(['date', 'source'])['mood'].value_counts().unstack(fill_value=0)
st.dataframe(summary)

# Additional Statistics
col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Total Posts Analyzed", len(filtered_df))

with col4:
    avg_mood = filtered_df['mood_score'].mean()
    st.metric("Average Mood Score", f"{avg_mood:.2f}")

with col5:
    most_common_mood = filtered_df['mood'].mode()[0]
    st.metric("Most Common Mood", most_common_mood)

# Download button for the data
st.download_button(
    label="Download Mood Summary CSV",
    data=summary.to_csv().encode('utf-8'),
    file_name='mood_summary.csv',
    mime='text/csv'
) 