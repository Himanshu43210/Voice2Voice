import os
import uuid
import sys
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import datetime
from playsound import playsound


# Add components to your system path and import the functions
sys.path.append('./components')
sys.path.append('./utils')

from mongo_db import MongoDB
from speech_to_text import transcribe_stream
from faiss_response_mapping import get_similar_response
from play_audio import play_audio_from_id, play_random_filler

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
conversation_id = generate_unique_id()
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

    opening_line = "Hello I'm Ishan from AryanTech Motors. We noticed that you were looking for some cars on our website. Are you interested in purchasing a new car at the moment?"
    print(opening_line)
    Intro = opening_line
    #play_audio_from_id('Intro')
    # fname = r"C:\Users\LENOVO\Documents\GitHub\Voice2VoiceAashi\components\audio_files\Intro.mp3"
    # playsound(fname)
    # play_audio_from_id('65086be7dea445ad95371510')

    while True:
        # Use the transcribe_stream function to get the query
        start_time = datetime.datetime.now()
        query = transcribe_stream()
        end_time = datetime.datetime.now()
        time_taken = (end_time - start_time).total_seconds()
        print(f'Time taken in STT: {time_taken}')

        play_random_filler()

        # Exit condition
        if query.lower() == 'exit':
            break

        context = ". ".join([entry[0] + ". " + entry[1] for entry in chat_history])
        full_query = context + ". " + query if context else query

        response = chain({"Chat History": full_query})

        end_time_langchain = datetime.datetime.now()
        time_taken_langchain = (end_time_langchain - end_time).total_seconds()
        print(f'Time taken in LangChain: {time_taken_langchain}')
        print(response)

        # Process the response from LangChain using get_similar_response
        matched_object_id, matched_response = get_similar_response(response['result'], conversation_id, query)
        end_time_faiss = datetime.datetime.now()
        time_taken_faiss = (end_time_faiss - end_time_langchain).total_seconds()
        print(f'Time taken in matching response: {time_taken_faiss}')
        print(matched_response)

        # play_audio_from_id(matched_object_id)

        chat_history.append((query, matched_response))

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
