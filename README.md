# CPSC 483 Machine Learning Project: Movie Recommender System 

Group Members: Anjali Patel, Amanda Shohdy, Connie Zhu || Fall 2024
_____________________________________________________________

Summary: Recommender systems have become prevalent in our daily lives and are integral components to various online platforms. In the streaming industry, such techniques provide a greater user interaction experience. This paper presents three commonly used models: collaborative filtering, content-based filtering, and lastly, a hybrid model, all generating personalized movie recommendations. Utilizing a Kaggle Dataset, we were able to gather sufficient data for this project: https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/data?select=rating.csv

Model/Technique Overview: Collaborative filtering leverages the ratings of similar users to make recommendations. Content-based filtering utilizes movie metadata such as titles and genres to recommend similar movies based on predicted content. Finally, the hybrid model encompasses a combination of the two filtering techniques, by normalizing and combining recommendation scores. 

Evaluation Metrics & Goal: To effectively measure each model on the same scale, Precision @k, Recall @k, F1-Score, Mean Average Precision, and Mean Squared Error have been computed for further analysis and rank the quality of the recommendations. Such rankings can be deemed experimental, as the goal of this paper is to integrate and evaluate multiple common recommendation techniques to provide valuable insights into the development of recommender systems. 


