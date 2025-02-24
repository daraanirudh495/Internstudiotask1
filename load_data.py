import pandas as pd

movies_df = pd.read_csv("C:\\Users\\Rahul Dara\\Downloads\\ml-100k\\ml-100k\\u.item", sep="|", encoding="latin-1", usecols=[0, 1], names=["movie_id", "title"])
print(movies_df.head())  # Check if movie data is loading correctly

# Define file paths
ratings_file = "C:\\Users\\Rahul Dara\\Downloads\\ml-100k\\ml-100k\\u.data"
movies_file = "C:\\Users\\Rahul Dara\\Downloads\\ml-100k\\ml-100k\\u.item"

# Load Ratings Data
ratings = pd.read_csv(
    ratings_file, 
    sep="\t", 
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Load Movies Data
movies = pd.read_csv(
    movies_file, 
    sep="|", 
    encoding="ISO-8859-1", 
    header=None, 
    usecols=[0, 1],
    names=["movie_id", "title"]
)

# Merge datasets on movie_id
df = pd.merge(ratings, movies, on="movie_id")

# Drop timestamp (not needed)
df.drop(columns=["timestamp"], inplace=True)

# Show dataset structure
print(df.head())

# Save processed dataset
df.to_csv("C:\\Users\\Rahul Dara\\internship\\data\\movies_ratings.csv", index=False)

print("Dataset loaded and saved successfully!")
