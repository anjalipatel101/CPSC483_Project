import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data():
    # Load in the dataset
    data = pd.read_csv("rating.csv")
    # Filter the rows where userId is less than or equal to 3000
    data = data[data['userId'] <= 7000]

    # Load in movie dataset
    movieData = pd.read_csv("movie.csv")

    # Create dictionary
    movieDict = movieData.set_index('movieId')[['title', 'genres']].to_dict(orient='index')

    return data, movieDict

def train_model(data):
    # Split data into training (70%) and testing (30%) sets
    trainingData, testingData = train_test_split(data, test_size=0.3, random_state=42)

    # Create user-item matrices for both training and test sets
    trainingMatrix = trainingData.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    testingMatrix = testingData.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # Fit the NearestNeighbors model on the training data
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(trainingMatrix)

    return knn, trainingMatrix, testingMatrix

# Evaluate the model by predicting ratings for users in the test set
def filteringRecommender(userId, n_recommendations=10):

    data, movieDict = load_data()
    knn, trainingMatrix, testingMatrix = train_model(data)

    # Ensure the user exists in the test set
    if userId not in testingMatrix.index:
        return f"User {userId} not found."

    # Select the target user
    targetUser = trainingMatrix.loc[[userId]] if userId in trainingMatrix.index else None
    if targetUser is None:
        return f"User {userId} not found."

    # Find similar users to the target user
    distances, indices = knn.kneighbors(targetUser, n_neighbors=3)
    similarUsers = indices.flatten()[1:]  # Exclude the target user themselves

    # Aggregate ratings from similar users for movies the target user hasn't rated
    similarUsersRatings = trainingMatrix.iloc[similarUsers]
    unratedMovies = trainingMatrix.loc[userId] == 0.0
    listOfMovies = similarUsersRatings.mean(axis=0)[unratedMovies]
    listOfMovies = listOfMovies.sort_values(ascending=False)
    recommendedMovies = listOfMovies.head(n_recommendations)
    # Convert series to list of pairs (movieId & rating)
    moviePairs = list(recommendedMovies.items())

    print(f"Recommended movies for user {userId}:\n")
    for index in range(len(moviePairs)) :
        print(f"{movieDict[moviePairs[index][0]]['title']:>85} : {movieDict[moviePairs[index][0]]['genres']}")

    # Calculate MSE for the recommended movies
    actual_ratings = testingMatrix.loc[userId, [movieId for movieId, _ in moviePairs]]
    predicted_ratings = [rating for _, rating in moviePairs]

    # Filter out movies that the user didn't rate in the test set
    valid_movies = actual_ratings[actual_ratings > 0].index
    actual_ratings = actual_ratings[valid_movies]
    predicted_ratings = [rating for movieId, rating in moviePairs if movieId in valid_movies]

    if len(actual_ratings) > 0 and len(predicted_ratings) > 0:
        mse = mean_squared_error(actual_ratings, predicted_ratings)
        print(f"\nMean Squared Error (MSE): {mse:.4f}")
    else:
        print("\nNo valid movies to calculate MSE.")

    # Precision, Recall, F1, and MAP
    actualItems = testingMatrix.loc[userId][testingMatrix.loc[userId] > 0].index.tolist()
    recommendedItems = [movieId for movieId, _ in moviePairs]

    precision = precisionAtK(recommendedItems, actualItems, n_recommendations)
    recall = recallAtK(recommendedItems, actualItems, n_recommendations)
    f1 = f1ScoreAtK(precision, recall)
    mapScore = meanAveragePrecision(recommendedItems, actualItems, n_recommendations)

    print(f"Precision @ {n_recommendations}: {precision:.4f}")
    print(f"Recall @ {n_recommendations}: {recall:.4f}")
    print(f"F1-Score @ {n_recommendations}: {f1:.4f}")
    print(f"MAP @ {n_recommendations}: {mapScore:.4f}")

def precisionAtK(recommended_items, actual_items, k):
    recommended_at_k = recommended_items[:k]
    relevant_at_k = set(recommended_at_k).intersection(set(actual_items))
    return len(relevant_at_k) / k if k > 0 else 0

def recallAtK(recommended_items, actual_items, k):
    recommended_at_k = recommended_items[:k]
    relevant_at_k = set(recommended_at_k).intersection(set(actual_items))
    return len(relevant_at_k) / len(actual_items) if len(actual_items) > 0 else 0

def f1ScoreAtK(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def meanAveragePrecision(recommended_items, actual_items, k):
    average_precision = 0
    relevant_count = 0
    for i, item in enumerate(recommended_items[:k], start=1):
        if item in actual_items:
            relevant_count += 1
            average_precision += relevant_count / i
    return average_precision / len(actual_items) if len(actual_items) > 0 else 0
