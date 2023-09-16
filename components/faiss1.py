from datetime import datetime
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
from mongo_db import MongoDB
from dotenv import load_dotenv
import os

load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
CSV_FILE_PATH = os.environ.get('CSV_FILE_PATH')
MONGO_DB_URI = os.environ.get('MONGO_DB_URI')
MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME')
MONGO_DB_COLLECTION = os.environ.get('MONGO_DB_COLLECTION')

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

mongo = MongoDB(MONGO_DB_URI, MONGO_DB_NAME) 



# Fetch the possible responses from MongoDB
cursor = MONGO_DB_COLLECTION.find({})
given_responses = [doc['response'] for doc in cursor]  # Assuming 'response' is the field name in MongoDB
object_ids = [str(doc['_id']) for doc in cursor]

vectorizer = TfidfVectorizer()
response_matrix = vectorizer.fit_transform(given_responses).toarray()

d = response_matrix.shape[1]

index = faiss.IndexFlatL2(d)
index.add(response_matrix.astype("float32"))

def find_most_similar(input_sentence):
    vectorized_sentence = vectorizer.transform([input_sentence]).toarray().astype("float32")
    distances, indices = index.search(vectorized_sentence, k=1)
    similarity_score = 1 / (1 + distances[0][0])

    return object_ids[indices[0][0]], similarity_score

def get_similar_response(input_sentence):
    start_time = datetime.now()

    matched_object_id, similarity_score = find_most_similar(input_sentence)

    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()  # Time taken in seconds

    print("Input Text:", input_sentence)
    print("Matched MongoDB ObjectID:", matched_object_id)
    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Time Taken: {time_taken:.4f} seconds")
    print("---------------------------------------------------------")
    
    return matched_object_id

# get_similar_response(input_sentence)
