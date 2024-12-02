import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

def load_data():
    # Load the dataset
    movie_data = pd.read_csv("movie.csv")

    # Combine relevant features for TF-IDF
    movie_data["combined_features"] = (
        movie_data["genres"].fillna("") + " " + movie_data["title"].fillna("")
    )
    return movie_data

def train_model(movie_data):
    # Create a TF-IDF matrix for combined features
    tfidf = TfidfVectorizer(stop_words="english")
    feature_matrix = tfidf.fit_transform(movie_data["combined_features"])

    # Calculate the cosine similarity matrix
    similarity_matrix = cosine_similarity(feature_matrix)
    return similarity_matrix

def contentFilteringRecommender(movie_title, n_recommendations=10):
    movie_data = load_data()
    similarity_matrix = train_model(movie_data)

    # Ensure the movie exists in the dataset
    movie_indices = movie_data[movie_data["title"].str.contains(movie_title, case=False)].index
    if len(movie_indices) == 0:
        return f"Movie '{movie_title}' not found."

    # Reference movie and similarity scores
    movie_index = movie_indices[0]
    reference_movie = movie_data.iloc[movie_index]
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    # Get top N recommendations
    similar_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations + 1]
    recommended_indices = [x[0] for x in similar_movies]
    similarity_values = [x[1] for x in similar_movies]

    # Extract movie details
    recommended_movies = movie_data.iloc[recommended_indices].copy()
    recommended_movies["similarity_score"] = similarity_values

    # Display recommendations
    print(f"\nTop {n_recommendations} Movies Similar to '{reference_movie['title']}':\n")
    for rank, (_, row) in enumerate(recommended_movies.iterrows(), 1):
        print(f"{rank:4d} | {row['title']:40s} | {row['genres']:20s} | {row['similarity_score']:.4f}")

    # Evaluate recommendations
    evaluate_metrics(recommended_movies, reference_movie, movie_data, similarity_values, n_recommendations)

def evaluate_metrics(recommended_movies, reference_movie, movie_data, predicted_ratings, k=10):
    # Relevant movies based on genres
    relevant_genres = reference_movie["genres"]
    relevant_movies = set(movie_data[movie_data["genres"].str.contains(relevant_genres, case=False, na=False)].index)
    recommended_indices = set(recommended_movies.head(k).index)

    # Metrics Calculation
    precision = precisionAtK(recommended_indices, relevant_movies, k)
    recall = recallAtK(recommended_indices, relevant_movies)
    f1 = f1ScoreAtK(precision, recall)
    map_score = meanAveragePrecision(recommended_indices, relevant_movies, k)

    # Generate pseudo-true ratings for MSE calculation (here based on similarity)
    actual_ratings = [1 if idx in relevant_movies else 0 for idx in recommended_indices]
    mse = mean_squared_error(actual_ratings, predicted_ratings[:len(actual_ratings)])

    print(f"\nPrecision @ {k}: {precision:.4f}")
    print(f"Recall @ {k}: {recall:.4f}")
    print(f"F1 Score @ {k}: {f1:.4f}")
    print(f"Mean Average Precision @ {k}: {map_score:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")

def precisionAtK(recommended_indices, relevant_movies, k):
    relevant_at_k = recommended_indices.intersection(relevant_movies)
    return len(relevant_at_k) / k if k > 0 else 0

def recallAtK(recommended_indices, relevant_movies):
    relevant_at_k = recommended_indices.intersection(relevant_movies)
    return len(relevant_at_k) / len(relevant_movies) if len(relevant_movies) > 0 else 0

def f1ScoreAtK(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def meanAveragePrecision(recommended_indices, relevant_movies, k):
    average_precision = 0
    relevant_count = 0
    for i, idx in enumerate(recommended_indices, start=1):
        if idx in relevant_movies:
            relevant_count += 1
            average_precision += relevant_count / i
    return average_precision / len(relevant_movies) if len(relevant_movies) > 0 else 0

# Run the content-based recommender
if __name__ == "__main__":
    movie_title = "Clueless"
    contentFilteringRecommender(movie_title)
