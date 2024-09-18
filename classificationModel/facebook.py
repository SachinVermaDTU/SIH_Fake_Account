import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reading Dataset
def read_data(file_path):
    """ Reads the dataset containing Facebook account information. """
    df = pd.read_csv(file_path)
    return df

# Feature Extraction
def extract_features(df):
    """ Extracts relevant features from the dataset for model training. """
    # Example feature extraction:
    df['num_friends'] = df['friends_list'].apply(lambda x: len(x.split(',')))  # Assuming friends_list is a comma-separated string
    df['num_posts'] = df['posts'].apply(lambda x: len(x.split('|')))  # Assuming posts is a pipe-separated string
    df['avg_post_length'] = df['posts'].apply(lambda x: np.mean([len(post) for post in x.split('|')]))

    # Returning relevant columns
    features = df[['num_friends', 'num_posts', 'avg_post_length']]
    return features

# Text Cleaning and Tokenization
def clean_and_tokenize_text(df):
    """ Cleans and tokenizes text for similarity analysis. """
    # Use TF-IDF Vectorizer for text tokenization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = vectorizer.fit_transform(df['posts'])
    return tfidf_matrix

# Jaccard Similarity Analysis
def calculate_jaccard_similarity(df):
    """ Calculates Jaccard similarity between wall posts of different accounts. """
    similarities = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            similarity = jaccard_score(df['posts'].iloc[i], df['posts'].iloc[j], average='binary')
            similarities.append((i, j, similarity))
    return similarities

# DBSCAN Clustering for Behavior Analysis
def cluster_behavior(features):
    """ Applies DBSCAN clustering to group accounts based on behavior. """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_features)
    return clusters

# Training and Evaluation (Placeholder for full model training, focus on DBSCAN and similarity)
def train_model(features, labels):
    """ Placeholder function to 'train' the model using DBSCAN for clustering. """
    clusters = cluster_behavior(features)
    return clusters


if __name__ == "__main__":
    # Step 1: Read Dataset
    df = read_data('facebook_data.csv')
    
    # Step 2: Extract Features
    features = extract_features(df)
    
    # Step 3: Clean and Tokenize Text for Jaccard Similarity
    tfidf_matrix = clean_and_tokenize_text(df)
    
    # Step 4: Calculate Jaccard Similarities
    jaccard_similarities = calculate_jaccard_similarity(df)
    
    # Step 5: Train Model (DBSCAN Clustering)
    clusters = train_model(features, df['label'])  # Assuming 'label' column exists for evaluation
    
    # Step 6: Output Results
    df['cluster'] = clusters
    df['suspicious'] = df['cluster'].apply(lambda x: 1 if x == -1 else 0)  # Marking noise points as suspicious
    
    # Print the flagged suspicious accounts
    suspicious_accounts = df[df['suspicious'] == 1]
    print(f"Suspicious accounts detected:\n{suspicious_accounts}")
    
    # Save results to a new CSV
    df.to_csv('fake_accounts_detected.csv', index=False)
