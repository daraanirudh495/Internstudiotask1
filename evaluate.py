import pickle
from collections import defaultdict

from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

# Load Model
with open("C:\\Users\\Rahul Dara\\internship\\models\\svd_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Dataset
ratings_file = "C:\\Users\\Rahul Dara\\internship\\data\\ml-100k\\u.data"
reader = Reader(line_format="user item rating timestamp", sep="\t")
data = Dataset.load_from_file(ratings_file, reader=reader)

# Split Data into Train and Test Set
trainset, testset = train_test_split(data, test_size=0.2)

# Get Predictions
predictions = model.test(testset)

# Function to Calculate Precision@K & Recall@K
def precision_recall_at_k(predictions, k=5, threshold=3.5):
    """Compute precision and recall at k for recommendation model."""
    user_est_true = defaultdict(list)
    
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    
    precisions = {}
    recalls = {}
    
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)  # Sort by estimated rating
        top_k = user_ratings[:k]

        # Compute precision and recall
        num_relevant = sum((true_r >= threshold) for (_, true_r) in top_k)
        num_total_relevant = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        precisions[uid] = num_relevant / k if k else 0
        recalls[uid] = num_relevant / num_total_relevant if num_total_relevant else 0

    # Average Precision & Recall
    avg_precision = sum(precisions.values()) / len(precisions)
    avg_recall = sum(recalls.values()) / len(recalls)

    return avg_precision, avg_recall

# Compute Precision and Recall
precision, recall = precision_recall_at_k(predictions, k=5, threshold=3.5)

# Print Results
print(f"Precision@5: {precision:.4f}")
print(f"Recall@5: {recall:.4f}")
