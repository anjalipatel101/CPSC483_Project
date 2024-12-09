# CPSC 483 Machine Learning Project: Movie Recommender System

Group Members: Anjali Patel, Connie Zhu, Amanda Shody 

Fall 2024

---
Recommender systems have become prevalent in our daily lives and are integral components to various online platforms. This repository contains the implementation of a movie recommendation system using the MovieLens 20M dataset. The project explores the following recommendation techniques and evaluates their effectiveness: content-based filtering, collaborative filtering, and a hybrid of both of the former methods.

### Prerequisites and How to Run

As the dataset was too large, it was not able to be uploaded to GitHub.

Please download the `rating.csv` dataset and ensure it is in the correct folder: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data?select=rating.csv

Ensure the path to the

To run this code, please execute `py main.py`, `python main.py`, or `python3 main.py` in a local terminal within the folder for this github repository.

### Dataset

Dataset: The MovieLens 20M dataset includes 20 million user ratings from 138493 users across 27278 movies, along with metadata about the movies such as genres and tags.

MovieLens 20M dataset: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data?select=rating.csv

### Features
Using `movie.csv` and `rating.csv`:

Recommendation Techniques:
- **Content-Based Filtering**: Recommends movies based on such as titles and genres to recommend similar movies based on predicted content.
- **Collaborative Filtering**: Recommends movies by analyzing user interaction patterns and preferences, leveraging the ratings of similar users to make recommendations
- **Hybrid Filtering**: Combination of the two filtering techniques, content-based and collaborative, by combining recommendation scores, normalizing, and using weight.
