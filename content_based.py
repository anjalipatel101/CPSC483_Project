import pandas as pd
import json
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_and_parse_json(json_str):
    # Replace single quotes with double quotes to make it valid JSON
    json_str = json_str.replace("'", "\"")
    return json.loads(json_str)

# Load data
movies_metadata = pd.read_csv('archive/movies_metadata.csv')
keywords = pd.read_csv('archive/keywords.csv')
credits = pd.read_csv('archive/credits.csv')
# links = pd.read_csv('archive/links.csv').dropna()
# ratings = pd.read_csv('archive/ratings_small.csv').dropna()

# Take a look
# print(movies_metadata['id'].head())
# print(keywords.head())
# print(credits.head())

# print(movies_metadata[movies_metadata['id'] == '862'])


# Ensure 'id' columns are in the same format (string type works well for merging)
movies_metadata['id'] = movies_metadata['id'].astype(str)
keywords['id'] = keywords['id'].astype(str)
credits['id'] = credits['id'].astype(str)

# print(movies_metadata[movies_metadata['id'] == '862'])
# print('keywords')
# print(keywords[keywords['id'] == '862'])
# print()
# print(credits[credits['id'] == '862'])

# dataframe
mkc_merge = movies_metadata.merge(keywords, on='id').merge(credits, on='id')
# if '862' in mkc_merge['id'].values:
#     print("Yes")
# else:
#     print("No")

print(mkc_merge.head())
print(len(mkc_merge))

# Function to parse JSON-like strings
def parse_features(x):
    try:
        return [i['name'] for i in ast.literal_eval(x)] # i is dictionary name
    except:
        return []

# Extract keywords
mkc_merge['keywords'] = mkc_merge['keywords'].apply(parse_features)
# Extract top 3 cast members for simplicity
mkc_merge['cast'] = mkc_merge['cast'].apply(lambda x: parse_features(x)[:3]) # lambda used to extract only 3

# Extract directors from crew
def get_directors(x):
    try:
        crew = ast.literal_eval(x)
        return [i['name'] for i in crew if i['job'] == 'Director']
    except:
        return []
mkc_merge['crew'] = mkc_merge['crew'].apply(get_directors)

# Extract genres
mkc_merge['genres'] = mkc_merge['genres'].apply(parse_features)

# Combine all text features into a single string
def combine_features(row):
    return ' '.join(row['overview'] if pd.notnull(row['overview']) else '') + ' ' + \
           ' '.join(row['keywords']) + ' ' + \
           ' '.join(row['cast']) + ' ' + \
           ' '.join(row['crew']) + ' ' + \
           ' '.join(row['genres'])

# Apply the function to create a 'content_profile' column
mkc_merge['content_profile'] = mkc_merge.apply(combine_features, axis=1) # column only, stored in column

# How it Works:
# Term Frequency (TF): Measures how frequently a word appears in a document.
# Inverse Document Frequency (IDF): Measures how rare a word is across the corpus.
# TF-IDF: The product of TF and IDF gives the final score for a word in a document. A high TF-IDF score means the word is important to that document.
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(mkc_merge['content_profile'])
# print(tfidf_matrix)

# Function to get movie recommendations based on cosine similarity
# Measures how similar two vectors are, based on their direction. A higher score means two movies share more similar content.
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# print(mkc_merge['title'].values)
# print(mkc_merge.columns) 
# if '862' in keywords['id'].values:
#     print("Yes")
# else:
#     print("No")

def get_recommendations(title, cosine_sim=cosine_sim):
    # Check if the movie exists
    if title not in mkc_merge['title'].values:
        print(f"Error: Movie title '{title}' not found in the dataset.")
        return []

    idx = mkc_merge[mkc_merge['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return mkc_merge['title'].iloc[movie_indices]

print(get_recommendations('Toy Story'))


# print(links.head())
# print(ratings.head())

# Function to safely parse JSON or handle already-parsed lists
# Function to handle parsing
# def parse_keywords(x):
#     # Check if entry is empty or null
#     if not x or x == "[]":
#         return ''
#     # Replace single quotes with double quotes for valid JSON format
#     if isinstance(x, str):
#         x = x.replace("'", "\"")
#     try:
#         # Attempt to parse JSON
#         data = json.loads(x)
#         # Join the 'name' field of each dictionary entry
#         return ' '.join([i['name'] for i in data])
#     except json.JSONDecodeError:
#         # Fall back to literal_eval for non-standard JSON format
#         try:
#             data = literal_eval(x)
#             return ' '.join([i['name'] for i in data])
#         except (ValueError, SyntaxError):
#             return ''  # Return empty if parsing still fails

# # Apply the function to the 'keywords' column
# keywords['keywords'] = keywords['keywords'].apply(parse_keywords)

# # Example: Parse keywords JSON column
# keywords['keywords'] = keywords['keywords'].apply(lambda x: ' '.join([i['name'] for i in json.loads(x)]))
# # credits['cast'] = credits['cast'].apply(lambda x: ' '.join([i['name'] for i in json.loads(x)[:3]])) # Top 3 cast
# # credits['crew'] = credits['crew'].apply(lambda x: ' '.join([i['name'] for i in json.loads(x) if i['job'] == 'Director']))


# print(keywords['keywords'])