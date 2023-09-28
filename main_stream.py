import os
import uuid
import sys
import datetime
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import StreamingRetrievalQA  # Import the streaming version
from langchain.llms import OpenAI
from langchain.callbacks import CallbackHandler  # Import CallbackHandler
from playsound import playsound

# Add components to your system path and import the functions
sys.path.append("./components")
sys.path.append("./utils")

from mongo_db import MongoDB
from speech_to_text import transcribe_stream
from faiss_response_mapping import get_similar_response
from play_audio import play_audio_from_id, play_random_filler

# Load environment variables from .env file
load_dotenv()

# Access environment variables
# ... (same as your original code)

mongo = MongoDB(MONGO_DB_URI, MONGO_DB_NAME)  # Initialize the MongoDB instance


def generate_unique_id():
    return str(uuid.uuid4())


conversation_id = generate_unique_id()


# Custom callback handler to process streamed messages
class AudioCallbackHandler(CallbackHandler):
    def handle_message(self, message):
        matched_object_id, matched_response = get_similar_response(
            message["result"], conversation_id, query
        )
        play_audio_from_id(matched_object_id)
        chat_history[-1] = (query, matched_response)


def chat_with_user():
    loader = CSVLoader(file_path=CSV_FILE_PATH)

    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])

    # Initialize the chain with streaming=True and provide the callback handler
    chain = StreamingRetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.vectorstore.as_retriever(),
        input_key="question",
        streaming=True,  # Enable streaming
        callbacks=[AudioCallbackHandler()],  # Provide the custom callback handler
    )

    opening_line = "Hello I'm Jenny from Gadget Hub. We noticed that you were looking for Google Pixel phone on our website. Are you interested in purchasing a new phone at the moment?"
    print(opening_line)
    play_audio_from_id("6511dcf3a21e2967d51e39d9")

    chat_history = [("Response", opening_line)]
    sales_bot_statement = "You are a sales bot. Your main objective is to convince the user to buy a Google Pixel phone rather than Iphone. Begin the conversation by discussing the benefits and features of Google Pixel. If the user shows interest in buying or knowing more, encourage them to visit the shop to experience the product firsthand. Be attentive to user's reactions and responses. If the user seems interested or willing to visit the shop, politely ask for their name and contact number to book an appointment for them. Say something like 'Can I have your name and contact number to schedule an appointment for you to visit our shop?'. Ensure to be courteous and maintain a friendly tone throughout the conversation, addressing any inquiries or concerns the user may have to facilitate the sales process. When they give you the name and number, end the conversation by telling then to have a great day.Give response to the last query and continue the conversation accordingly"

    while True:
        start_time = datetime.datetime.now()
        query = transcribe_stream()  # Getting the user's query.

        if query.lower() == "exit":
            break

        # Create structured context with separate indications for user queries and responses.
        context = ". ".join(
            ["Query: " + entry[0] + ". Response: " + entry[1] for entry in chat_history]
        )

        # Append current query to chat_history before determining the last_query.
        chat_history.append(
            (query, "")
        )  # Temporary append, will update the response later.

        # Now determine the last query, which is the current query.
        last_query = '"Query": ' + " ".join(['"' + query + '"' for _ in range(3)])

        # Combining the context with the emphasized last query to form the full_query.
        full_query = sales_bot_statement + (
            context + ". " + last_query if context else last_query
        )

        # Now, the response is streamed and handled by the callback handler
        chain({"question": full_query})


def save_chat_to_txt(filename, chat_history):
    with open(filename, "w") as file:
        for entry in chat_history:
            user_query, bot_response = entry
            file.write(f"User: {user_query}\nBot: {bot_response}\n\n")


if __name__ == "__main__":
    chat_history = chat_with_user()
    save_chat_to_txt("chat_history.txt", chat_history)
    mongo.close_connection()  # Close the MongoDB connection
