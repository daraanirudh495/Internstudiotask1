import pickle

import pandas as pd

# Load Models
with open("C:\\Users\\Rahul Dara\\internship\\models\\svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)

with open("C:\\Users\\Rahul Dara\\internship\\models\\vectorizer.pkl", "rb") as f:
    vectorizer, cosine_sim = pickle.load(f)

# Load Movie Data
movies_file = "C:\\Users\\Rahul Dara\\Downloads\\ml-100k\\ml-100k\\u.item"
movies = pd.read_csv(movies_file, sep="|", encoding="ISO-8859-1", header=None, usecols=[0, 1])
movies.columns = ["movie_id", "title"]

# Function to Get Collaborative Recommendations
def get_collaborative_recommendations(user_id, n=5):
    all_movie_ids = movies["movie_id"].tolist()
    predictions = [svd_model.predict(user_id, iid) for iid in all_movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_n = predictions[:n]
    top_movie_ids = [pred.iid for pred in top_n]
    return movies[movies["movie_id"].isin(top_movie_ids)]

# Function to Get Content-Based Recommendations
def get_content_based_recommendations(movie_title, n=5):
    idx = movies[movies["title"] == movie_title].index[0]
    similar_movies = list(enumerate(cosine_sim[idx]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in similar_movies]
    return movies.iloc[movie_indices]

# Hybrid Recommendation (Weighted Combination)
def get_hybrid_recommendations(user_id, movie_title, alpha=0.5, n=5):
    collab_recs = get_collaborative_recommendations(user_id, n)
    content_recs = get_content_based_recommendations(movie_title, n)
    
    collab_recs["score"] = alpha * 1.0
    content_recs["score"] = (1 - alpha) * 1.0
    
    hybrid_recs = pd.concat([collab_recs, content_recs]).sort_values(by="score", ascending=False)
    return hybrid_recs.drop(columns=["score"]).head(n)
