import pickle

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Load Dataset
ratings_file ="C:\\Users\\Rahul Dara\\Downloads\\ml-100k\\ml-100k\\u.data"
reader = Reader(line_format="user item rating timestamp", sep="\t")
data = Dataset.load_from_file(ratings_file, reader=reader)

# Train-Test Split
trainset, testset = train_test_split(data, test_size=0.2)

# Train Model (SVD - Collaborative Filtering)
model = SVD()
model.fit(trainset)

# Save Model
with open("C:\\Users\\Rahul Dara\\internship\\models\\svd_model.pkl", "wb") as f:
    pickle.dump(model,f)

print("Collaborative Filtering Model (SVD) trained and saved!")
