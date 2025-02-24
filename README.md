# MovieLens 100K Recommendation System

This project implements a **Recommendation System** using the **MovieLens 100K Dataset** to build personalized movie recommendations. It leverages **Collaborative Filtering** (SVD) and **Content-Based Filtering** techniques to recommend movies based on user preferences.

## **Project Structure**
. ├── data/ # Folder to store the datasets │ └── ml-100k/ # MovieLens 100K dataset │ ├── u.data # User ratings dataset │ ├── u.item # Movie metadata │ ├── u.genre # Movie genres │ └── u.user # User demographic information ├── models/ # Folder to save trained models │ └── svd_model.pkl # Saved collaborative filtering model (SVD) ├── src/ # Source code for the project │ ├── collaborative.py # Collaborative filtering model (SVD) │ ├── content_based.py # Content-based filtering model │ ├── hybrid.py # Hybrid recommendation system │ ├── load_data.py # Script to load and preprocess data │ └── evaluation.py # Evaluate recommendation models ├── requirements.txt # Python dependencies └── README.md # Project documentation
