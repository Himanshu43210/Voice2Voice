import os
import uuid
import sys
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


# Add components to your system path and import the functions
sys.path.append('./components')
sys.path.append('./utils')

from mongo_db import MongoDB
from speech_to_text import transcribe_stream
from faiss_response_mapping import get_similar_response
from play_audio import play_audio_from_id

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
CSV_FILE_PATH = os.environ.get('CSV_FILE_PATH')
MONGO_DB_URI = os.environ.get('MONGO_DB_URI')
MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME')
MONGO_DB_COLLECTION = os.environ.get('MONGO_DB_COLLECTION')

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

mongo = MongoDB(MONGO_DB_URI, MONGO_DB_NAME)  # Initialize the MongoDB instance

def generate_unique_id():
    return str(uuid.uuid4())

def chat_with_user():
    loader = CSVLoader(file_path=CSV_FILE_PATH)

    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])

    chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.vectorstore.as_retriever(),
        input_key="question"
    )

    chat_history = []

    opening_line = "Hello, I'm Jacob from AryanTech Company. I'm calling you regarding our service to assist you in securing a job. Are you looking for any job opportunities right now?"
    print(opening_line)

    while True:
        # Use the transcribe_stream function to get the query
        query = transcribe_stream()

        # Exit condition
        if query.lower() == 'exit':
            break

        context = ". ".join([entry[0] + ". " + entry[1] for entry in chat_history])
        full_query = context + ". " + query if context else query

        response = chain({"question": full_query})

        # Process the response from LangChain using get_similar_response
        matched_object_id, matched_response = get_similar_response(response['result'])
        play_audio_from_id(matched_object_id)

        chat_history.append((query, matched_response))

        # Save the response to MongoDB
        conversation_id = generate_unique_id()
        mongo.insert_response(matched_response, conversation_id)

        print(matched_response)

    return chat_history

def save_chat_to_txt(filename, chat_history):
    with open(filename, 'w') as file:
        for entry in chat_history:
            user_query, bot_response = entry
            file.write(f"User: {user_query}\nBot: {bot_response}\n\n")

if __name__ == "__main__":
    chat_history = chat_with_user()
    save_chat_to_txt('chat_history.txt', chat_history)
    mongo.close_connection()  # Close the MongoDB connection
