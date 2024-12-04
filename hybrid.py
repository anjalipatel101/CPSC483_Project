# Using Anjali's and Amanda's
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from collaborativeFiltering import precisionAtK, recallAtK, f1ScoreAtK, meanAveragePrecision
from sklearn.model_selection import train_test_split

def load_data():
    ratings = pd.read_csv("/Users/conneigh8/Documents/CPSC483_FALL2024/CPSC483_Project/MovieLens20M/rating.csv")
    movies = pd.read_csv("/Users/conneigh8/Documents/CPSC483_FALL2024/CPSC483_Project/MovieLens20M/movie.csv")

    # # Filter for active users (users with at least 50 ratings)
    # active_users = ratings['userId'].value_counts()[ratings['userId'].value_counts() > 50].index
    # # Filter for popular movies (movies with at least 50 ratings)
    # popular_movies = ratings['movieId'].value_counts()[ratings['movieId'].value_counts() > 50].index
    # # Filter for ratings >= 4
    # ratings = ratings[(ratings['userId'].isin(active_users)) &
    #                         (ratings['movieId'].isin(popular_movies)) &
    #                         (ratings['rating'] >= 4)]
    ratings = ratings[ratings['userId'] <= 7000]

    return ratings, movies

# ----- Content-Based Filtering (CBF) -----
class ContentBasedRecommender:
    def __init__(self, movies):
        self.movies = movies
        self._prepare_feature_matrix()

    def _prepare_feature_matrix(self):
        self.movies['combined_features'] = self.movies['genres'] + ' ' + self.movies['title']
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.feature_matrix = self.tfidf.fit_transform(self.movies['combined_features'].fillna(''))
        self.similarity_matrix = cosine_similarity(self.feature_matrix)

    def recommend(self, liked_movie_ids, n_recommendations=5):
        scores = np.zeros(len(self.movies))
        for movie_id in liked_movie_ids:
            if movie_id not in self.movies['movieId'].values:
                continue
            index = self.movies[self.movies['movieId'] == movie_id].index[0]
            scores += self.similarity_matrix[index]
        
        recommended_indices = scores.argsort()[-n_recommendations:][::-1]
        recommended_movies = self.movies.iloc[recommended_indices]

        recommended_movies = recommended_movies[~recommended_movies['movieId'].isin(liked_movie_ids)]
        recommended_movies = self.movies.iloc[recommended_indices].copy()  # Make a copy
        recommended_movies.loc[:, 'similarity_score'] = scores[recommended_indices]  # Using .loc to add the similarity_score column
        return recommended_movies[['movieId', 'title', 'genres', 'similarity_score']]

# ----- Collaborative Filtering (CF) -----
def collaborative_filtering(user_id, user_ratings, n_recommendations=5, n_neighbors=6):
    user_movie_matrix = user_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_movie_matrix)
    if user_id not in user_movie_matrix.index:
        return None  # User not in training data
    distances, indices = knn.kneighbors([user_movie_matrix.loc[user_id]], n_neighbors=n_neighbors)
    similar_users = indices.flatten()[1:]
    similar_ratings = user_movie_matrix.iloc[similar_users].mean(axis=0)
    unrated_movies = user_movie_matrix.loc[user_id] == 0
    return similar_ratings[unrated_movies].sort_values(ascending=False).head(n_recommendations)

# ----- Hybrid Approach -----
def hybrid_recommender(user_id, ratings, movies, n_recommendations=5, cbf_weight=0.5, cf_weight=0.5, n_neighbors=5):
    # Content-based recommendations
    user_liked_movies = ratings[ratings['userId'] == user_id].sort_values(by='rating', ascending=False)['movieId'].head(5).tolist()
    # Initialize CBF
    cbf_recommender = ContentBasedRecommender(movies)
    # Exclude movies already rated by the user
    cbf_recs = cbf_recommender.recommend(user_liked_movies, n_recommendations)

    # Collaborative recommendations
    cf_recs = collaborative_filtering(user_id, ratings, n_recommendations, n_neighbors)
    
    if cf_recs is None:
        print(f"No CF recommendations available for User {user_id}.")
        return cbf_recs  # Fall back to CBF
    # cbf_recs = cbf_recs[~cbf_recs['movieId'].isin(user_liked_movies)]

    def normalize_scores(scores):
        if scores.empty or scores.min() == scores.max():
            return scores  # Return unchanged if all scores are the same
        return (scores - scores.min()) / (scores.max() - scores.min())

    # Normalize Scores
    cbf_recs['normalized_similarity'] = normalize_scores(cbf_recs['similarity_score'])

    cf_recs = cf_recs.reset_index().rename(columns={0: 'score', 'index': 'movieId'})
    cf_recs['normalized_score'] = normalize_scores(cf_recs['score'])

    # Merge Results
    merged = pd.merge(cbf_recs, cf_recs, on='movieId', how='outer').fillna(0)
    merged['final_score'] = (
        cbf_weight * merged['normalized_similarity'] +
        cf_weight * merged['normalized_score']
    )
    
    # Add movie details
    merged = pd.merge(merged, movies[['movieId', 'title', 'genres']], on='movieId', how='left')
    merged = merged.drop(columns=['title_x', 'genres_x', 'title_y', 'genres_y'], errors='ignore')
    merged = pd.merge(merged, movies[['movieId', 'title', 'genres']], on='movieId', how='left')
    # merged = merged.sort_values(by='final_score', ascending=False).head(n_recommendations)
    # print(merged[['title', 'genres', 'final_score']])

    # Sort and Return Top Recommendations
    return merged.sort_values(by='final_score', ascending=False).head(n_recommendations)

# user_id = 1  
# recommendations = hybrid_recommender(user_id)
# print(recommendations[['movieId', 'title', 'genres']])

def split_data(ratings, test_size=0.3):
    # Split dataset into training and testing sets (no overlap)
    train_data, test_data = train_test_split(ratings, test_size=test_size, random_state=42)
    return train_data, test_data

# Evaluation function for the hybrid recommender
def evaluate_hybrid_recommender(user_id, ratings, movies, hybrid_recommender, n_recommendations=5):
    train_data, test_data = split_data(ratings)
    # Actual ratings for the user in the test set
    actual_ratings = test_data[test_data['userId'] == user_id].set_index('movieId')['rating']

    # Get recommendations from the hybrid system
    recommendations = hybrid_recommender(user_id, train_data, movies, n_recommendations=n_recommendations)

    filtered_recommendations = [movie_id for movie_id in recommendations['movieId'] if movie_id in actual_ratings.index]

    # Prepare predicted ratings from the recommended movies
    predicted_ratings = []
    for movie_id in filtered_recommendations:
        # if movie_id in actual_ratings.index:
            predicted_ratings.append(recommendations.loc[recommendations['movieId'] == movie_id, 'final_score'].values[0])
        # else:
        #     # If the user hasn't rated this movie in the test set, we can't predict its rating
        #     predicted_ratings.append(0)  # or some default value
    
    predicted_ratings = [score * 4 + 1 for score in predicted_ratings]

    # Ensure both actual and predicted ratings have the same length
    actual_ratings_filtered = actual_ratings[filtered_recommendations]
        
    # print(len(actual_ratings_filtered))
    # print(len(predicted_ratings))
    # print(actual_ratings_filtered.unique())
    # print(set(predicted_ratings))

    if len(actual_ratings) > 0 and len(predicted_ratings) > 0:
        mse = mean_squared_error(actual_ratings_filtered, predicted_ratings)
        print(f"\nMean Squared Error (MSE): {mse:.4f}")
    else:
        print("\nNo valid movies to calculate MSE.")
    
    # Precision, Recall, F1, and MAP
    actual_items = actual_ratings.index.tolist()
    recommended_items = recommendations['movieId'].tolist()
    # print(len(actual_items))
    # print(len(recommended_items))

    precision = precisionAtK(recommended_items, actual_items, n_recommendations)
    recall = recallAtK(recommended_items, actual_items, n_recommendations)
    f1 = f1ScoreAtK(precision, recall)
    map_score = meanAveragePrecision(recommended_items, actual_items, n_recommendations)

    print(f"Precision @ {n_recommendations}: {precision:.4f}")
    print(f"Recall @ {n_recommendations}: {recall:.4f}")
    print(f"F1-Score @ {n_recommendations}: {f1:.4f}")
    print(f"MAP @ {n_recommendations}: {map_score:.4f}")

# evaluate_hybrid_recommender(user_id, hybrid_recommender, ratings, 10)

# THIS IS IT in test.py
# Mean Squared Error (MSE): 7.2418
# Precision @ 10: 0.4000
# Recall @ 10: 0.0769
# F1-Score @ 10: 0.1290
# MAP @ 10: 0.0456

# Mean Squared Error (MSE): 4.6404
# Precision @ 10: 0.4000
# Recall @ 10: 0.0769
# F1-Score @ 10: 0.1290
# MAP @ 10: 0.0448