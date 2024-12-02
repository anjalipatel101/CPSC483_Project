# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oUjITBDPki7jZWTDHqKi44eE9Q2xaRrz
"""

from collaborativefiltering import filteringRecommender
from hybrid import hybrid_recommender, evaluate_hybrid_recommender, load_data

def runRecommendation(choice):
    if (choice == 2):
        print("Enter your userId: ", end=' ')
        userId = int(input())
        filteringRecommender(userId)
    if (choice == 3):
        print("Enter your userId: ", end=' ')
        userId = int(input())
        recommendations = hybrid_recommender(userId, n_recommendations=10)
        print(recommendations[['movieId', 'title', 'genres']])
        ratings, movies = load_data()
        evaluate_hybrid_recommender(userId, hybrid_recommender, ratings, n_recommendations=10)

if __name__ == "__main__":
    print("How would you like to generate movies?")
    print("     1. Find similar movies to your favorite")
    print("     2. Movies tailored to you")
    print("     3. Hybrid approach")
    print("Enter your choice: ", end=' ')
    choice = int(input())
    runRecommendation(choice)
    print()