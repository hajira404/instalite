# --- 📊 Instagram Mood Analyzer ---
import json
import os
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pprint import pprint
import pandas as pd
import numpy as np

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# --- 🔄 General Sentiment Analyzer Function ---
def analyze_sentiment(text):
    if not text: return {"compound": 0.0, "label": "neutral"}
    scores = sia.polarity_scores(text)
    c = scores["compound"]
    # More sensitive thresholds for mood detection
    if c > 0.3:  # Changed from 0.5
        label = "happy"
    elif c < -0.3:  # Changed from -0.5
        label = "sad"
    else:
        label = "neutral"
    return {"compound": c, "label": label}

# --- 📂 Load All Files from Folder ---
folder_path = "data" 
all_data = []

# Likes
with open(os.path.join(folder_path, "insta_mood_mock_likes.json")) as f:
    data = json.load(f)
    for post in data:
        post['source'] = 'likes'
        post['timestamp'] = datetime.fromisoformat(post['timestamp'])
        post['date'] = post['timestamp'].date()
        s = analyze_sentiment(post.get('caption', ''))
        post['mood'] = s['label']
        post['mood_score'] = s['compound']
        all_data.append(post)

# Reels
with open(os.path.join(folder_path, "insta_mood_mock_reels.json")) as f:
    data = json.load(f)
    for post in data:
        post['source'] = 'reels'
        post['timestamp'] = datetime.fromisoformat(post['timestamp'])
        post['date'] = post['timestamp'].date()
        s = analyze_sentiment(post.get('caption', ''))
        post['mood'] = s['label']
        post['mood_score'] = s['compound']
        all_data.append(post)

# Messages
with open(os.path.join(folder_path, "insta_mood_mock_messages.json")) as f:
    data = json.load(f)
    for convo in data:
        if 'messages' not in convo:
            continue
        for msg in convo['messages']:
            msg['source'] = 'messages'
            msg['timestamp'] = datetime.fromisoformat(msg['timestamp'])
            msg['date'] = msg['timestamp'].date()
            s = analyze_sentiment(msg.get('content', ''))
            msg['mood'] = s['label']
            msg['mood_score'] = s['compound']
            all_data.append(msg)

# Ads and Interests data
with open(os.path.join(folder_path, "insta_mood_mock_ads_and_interests.json")) as f:
    data = json.load(f)
    for ad in data:
        if not isinstance(ad, dict):
            continue
        
        ad['timestamp'] = datetime.fromisoformat(ad['timestamp'])
        ad['date'] = ad['timestamp'].date()
        ad['category'] = ad.get('category', 'Unknown')
        ad['confidence'] = ad.get('confidence', 0.0)
        
        category_to_content = {
            "Fashion": "New trends in fashion this season.",
            "Travel": "Plan your next adventure today!",
            "AI & Tech": "Explore the future of AI and technology.",
            "Breakup Recovery": "Focus on healing and self-care.",
            "Mental Wellness": "Take care of your mental health today."
        }
        content = category_to_content.get(ad['category'], "Stay positive!")
        s = analyze_sentiment(content)
        ad['mood'] = s['label']
        ad['mood_score'] = s['compound']
        all_data.append(ad)

# Video Watch History
with open(os.path.join(folder_path, "insta_mood_mock_watch_history.json")) as f:
    data = json.load(f)
    for vid in data:
        vid['source'] = 'video_watch'
        vid['timestamp'] = datetime.fromisoformat(vid['timestamp'])
        vid['date'] = vid['timestamp'].date()
        vid['mood'] = "neutral"
        vid['mood_score'] = 0.0
        all_data.append(vid)

# --- 📊 Convert to DataFrame for Summary ---
df = pd.DataFrame(all_data)
summary = df.groupby(['date', 'source'])['mood'].value_counts().unstack(fill_value=0)
print("📅 Mood Summary Table:")
print(summary)

# Save it to a CSV
summary.to_csv("mood_summary.csv")

# --- 📉 Enhanced Anomaly Detection ---
# Group by date: average mood score per day
daily_mood = df.groupby('date')['mood_score'].mean().reset_index()
daily_mood['mood_change'] = daily_mood['mood_score'].diff()

# More sensitive anomaly detection
# Flag days where mood drops more than 0.3 (changed from 0.5)
daily_mood['anomaly'] = daily_mood['mood_change'].apply(lambda x: True if x is not None and x < -0.3 else False)

# Additional anomaly conditions
daily_mood['volatility'] = df.groupby('date')['mood_score'].std()
daily_mood['activity_level'] = df.groupby('date').size()

# Flag high volatility days
daily_mood['volatility_anomaly'] = daily_mood['volatility'] > 0.4

# Flag low activity days with negative mood
daily_mood['low_activity_anomaly'] = (daily_mood['activity_level'] < 3) & (daily_mood['mood_score'] < -0.2)

# Show all types of anomalies
print("\n📉 Mood Anomalies Detected:")
print("=" * 50)

# Sudden mood drops
mood_drops = daily_mood[daily_mood['anomaly']]
if not mood_drops.empty:
    print("\n🚨 Sudden Mood Drops:")
    for _, row in mood_drops.iterrows():
        print(f"Date: {row['date']}")
        print(f"Mood dropped from {row['mood_score'] - row['mood_change']:.2f} to {row['mood_score']:.2f}")
        print(f"Change: {row['mood_change']:.2f}")
        print("-" * 30)

# High volatility days
volatility_anomalies = daily_mood[daily_mood['volatility_anomaly']]
if not volatility_anomalies.empty:
    print("\n📊 High Mood Volatility Days:")
    for _, row in volatility_anomalies.iterrows():
        print(f"Date: {row['date']}")
        print(f"Volatility: {row['volatility']:.2f}")
        print(f"Average Mood: {row['mood_score']:.2f}")
        print("-" * 30)

# Low activity with negative mood
low_activity_anomalies = daily_mood[daily_mood['low_activity_anomaly']]
if not low_activity_anomalies.empty:
    print("\n⚠️ Low Activity with Negative Mood:")
    for _, row in low_activity_anomalies.iterrows():
        print(f"Date: {row['date']}")
        print(f"Activity Level: {row['activity_level']}")
        print(f"Mood Score: {row['mood_score']:.2f}")
        print("-" * 30)

if mood_drops.empty and volatility_anomalies.empty and low_activity_anomalies.empty:
    print("No significant anomalies detected in the analyzed period.")
