import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Movie Data
movies_file = "C:\\Users\\Rahul Dara\\Downloads\\ml-100k\\ml-100k\\u.item"

movies = pd.read_csv(movies_file, sep="|", encoding="ISO-8859-1", header=None, usecols=[0, 1])
movies.columns = ["movie_id", "title"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["title"])

# Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Save Model
with open("C:\\Users\\Rahul Dara\\internship\\models\\vectorizer.pkl", "wb") as f:
    pickle.dump((vectorizer, cosine_sim), f)

print("Content-Based Filtering Model trained and saved!")
