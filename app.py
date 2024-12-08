from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
movies = pd.read_csv('movies.csv')  # Dataset contains 'title' and 'genres'

# Preprocessing
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Helper function to recommend movies
def recommend_movies(title, num_recommendations=10):
    if title not in movies['title'].values:
        return ["Movie not found. Try another one."]
    
    idx = movies[movies['title'] == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in similarity_scores[1:num_recommendations + 1]]
    return movies['title'].iloc[recommended_indices].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('movieName')
    recommendations = recommend_movies(user_input)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
