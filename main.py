import streamlit as st
import joblib
from joblib import load
import pandas as pd
import numpy as np 
import os
from surprise import Dataset, Reader 

# Debugging: Check if files exist
if not os.path.exists('movie_recommender.joblib'):
    st.error("File 'movie_recommender.joblib' not found.")
    st.stop()

if not os.path.exists('titles.xls'):
    st.error("File 'titles.xls' not found.")
    st.stop()

if not os.path.exists('user_interactions.xls'):
    st.error("File 'user_interactions.xls' not found.")
    st.stop()

# Load the saved model and data
try:
    model = load('movie_recommender.joblib')  # Your trained SVD model
    titles = pd.read_excel('titles.xls', usecols=['id', 'title', 'genres', 'release_year'])  # Movie metadata
    user_interactions = pd.read_excel('user_interactions.xls', usecols=['user_id', 'id', 'rating'])  # User interactions
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Streamlit App
st.title("Movie Recommendation System")

# User Input
user_id = st.number_input("Enter your User ID", min_value=1, max_value=10000, value=1)
genre_filter = st.text_input("Enter a genre to filter (optional):").strip()  # Remove leading/trailing spaces
year_filter = st.number_input("Enter a release year to filter (optional):", min_value=1900, max_value=2023, value=None)

# Recommendation Function
def recommend_for_user(user_id, genre_filter=None, year_filter=None):
    # Get movies the user has already rated
    user_rated_movies = user_interactions[user_interactions['user_id'] == user_id]['id'].tolist()
    
    # Filter movies based on genre and year
    filtered_movies = titles.copy()
    
    if genre_filter:
        filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(genre_filter, case=False, na=False)]
    
    if year_filter:
        filtered_movies = filtered_movies[filtered_movies['release_year'] == year_filter]
    
    if filtered_movies.empty:
        return [{"title": "No movies found matching your criteria.", "rating": None}]
    
    # Get unrated movies
    unrated_movies = filtered_movies[~filtered_movies['id'].isin(user_rated_movies)]
    
    if unrated_movies.empty:
        return [{"title": "No new recommendations available.", "rating": None}]
    
    # Predict ratings for unrated movies
    predictions = []
    for movie_id in unrated_movies['id'].unique():
        pred = model.predict(user_id, movie_id)
        movie_title = unrated_movies[unrated_movies['id'] == movie_id]['title'].values[0]
        predictions.append({"title": movie_title, "rating": pred.est})
    
    # Sort by predicted rating (descending)
    predictions.sort(key=lambda x: x['rating'], reverse=True)
    
    return predictions[:10]  # Return top 10 recommendations

# Get Recommendations
if st.button("Get Recommendations"):
    recommendations = recommend_for_user(user_id, genre_filter, year_filter)
    
    st.write("### Top Recommendations:")
    for movie in recommendations:
        if movie['rating'] is not None:
            st.write(f"**{movie['title']}** (Predicted Rating: {movie['rating']:.2f})")
        else:
            st.write(f"**{movie['title']}**")
