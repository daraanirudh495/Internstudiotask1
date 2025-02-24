import pickle

import pandas as pd
from flask import Flask, render_template, request
from surprise import Dataset, Reader

app = Flask(__name__)

# Load Model
with open("C:\\Users\\Rahul Dara\\internship\\models\\svd_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Movies Data
movies_df = pd.read_csv("C:\\Users\\Rahul Dara\\internship\\data\\ml-100k\\u.item", sep="|", encoding="latin-1", usecols=[0, 1], names=["movie_id", "title"])


# Function to Get Recommendations
def get_recommendations(user_id, k=5):
    all_movie_ids = movies_df["movie_id"].tolist()
    predictions = [model.predict(user_id, movie_id) for movie_id in all_movie_ids]
    
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_k = predictions[:k]
    
    recommended_movies = [movies_df[movies_df["movie_id"] == int(pred.iid)]["title"].values[0] for pred in top_k]
    return recommended_movies

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = int(request.form["user_id"])
    recommendations = get_recommendations(user_id)
    return render_template("index.html", movies=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
