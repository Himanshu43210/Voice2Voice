import pandas as pd
from datetime import datetime
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# Read the CSV into a DataFrame
df = pd.read_csv('contextnew.csv')

queries = df['Query'].tolist()
given_responses = df['Response'].tolist()

vectorizer = TfidfVectorizer()
response_matrix = vectorizer.fit_transform(given_responses).toarray()


# Define the dimensionality of the vectors
d = response_matrix.shape[1]

# Create a FAISS index for the given responses
index = faiss.IndexFlatL2(d)
index.add(response_matrix.astype('float32'))


def find_most_similar(input_sentence):
    vectorized_sentence = vectorizer.transform([input_sentence]).toarray().astype('float32')
    
    # Search the index for the closest vector
    distances, indices = index.search(vectorized_sentence, k=1)
    
    # Convert the negative L2 distance to a positive score for better interpretation
    similarity_score = -distances[0][0]
    
    return given_responses[indices[0][0]], similarity_score

input_sentences = [
    "we provide client testimonials",
    # Add more sentences as required
]

for input_sentence in input_sentences:
    start_time = datetime.now()

    response, similarity_score = find_most_similar(input_sentence)

    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()  # Time taken in seconds

    print("Input Text:", input_sentence)
    print("Output Text:", response)
    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Time Taken: {time_taken:.4f} seconds")
    print("---------------------------------------------------------")
