import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()
MONGO_DB_URI = os.environ.get('MONGO_DB_URI')  
MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME') 

# Connect to MongoDB
client = MongoClient(MONGO_DB_URI)
db = client[MONGO_DB_NAME]

def query_mismatched_responses():
    # Query the collection for documents where matched_text and reference_text are not the same
    mismatched_docs = db.responses_ext1.find({
        "$expr": {
            "$ne": ["$matched_text", "$reference_text"]
        }
    })

    # List to store mismatched documents
    mismatched_list = []

    # Display the specified fields for each document and add to the list
    for doc in mismatched_docs:
        print("Response:", doc.get("response"))
        print("Matched Text:", doc.get("matched_text"))
        print("Reference Text:", doc.get("reference_text"))
        print("Similarity Score:", doc.get("similarity_score"))
        print("Time Taken:", doc.get("time_taken"))
        print("-" * 50)  # Separator for better readability

        mismatched_list.append(doc)

    # Insert mismatched documents into the mismatched_responses collection
    if mismatched_list:
        db.mismatched_responses.insert_many(mismatched_list)

    # Close the MongoDB connection
    client.close()

# Call the method
if __name__ == "__main__":
    query_mismatched_responses()
