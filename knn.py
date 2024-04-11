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
     

def calculate_similarity(reference_sentence, target_sentence):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Vectorize the reference sentence and the target sentence
    sentence_vectors = vectorizer.fit_transform([reference_sentence, target_sentence])
    
    # Calculate cosine similarity between the vectors
    similarity_score = cosine_similarity(sentence_vectors[0], sentence_vectors[1])[0][0]
    
    return similarity_score

# Example usage:
reference_sentence = sentences[0]
target_sentence = new_sentence

# Calculate similarity score between the reference and target sentences
similarity_score = calculate_similarity(reference_sentence, target_sentence)


print(f"Similarity Score: {similarity_score}")
