from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Sample sentences (you can replace these with your dataset)
sentences = [
    "I am feeling anxious and stressed today.",
    "Mental health is important for overall well-being.",
    "Depression can be a serious mental health condition.",
    "Yoga and meditation help in reducing stress.",
    "I feel a sense of calm after talking to my therapist."
]

# New sentence to compare
new_sentence = "Managing stress is crucial for good mental health."

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(sentences)
new_vec = vectorizer.transform([new_sentence])

# KNN model
k = 3  # Number of neighbors to consider
knn = NearestNeighbors(n_neighbors=k, metric='cosine')
knn.fit(X)

# Find nearest neighbors for the new sentence
distances, indices = knn.kneighbors(new_vec)

# Print the most similar sentences
print(f"Top {k} most similar sentences:")
for i in range(k):
    print(sentences[indices[0][i]])
