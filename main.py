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


    opening_line = "Hello I'm Jenny from Gadget Hub. We noticed that you were looking for Google Pixel phone on our website. Are you interested in purchasing a new phone at the moment?"
    print(opening_line)
    play_audio_from_id('6511dcf3a21e2967d51e39d9')
    
    chat_history = [('Response', opening_line)]
    #sales_bot_statement = 'You are a sales bot and you are having a conversation with a user, and your task is to convince the user to buy Google pixel. Also ask the user to visit your shop to try it. Ask for their name and contact number to confirm the appointment. Respond to the user accordingly. '
    #sales_bot_statement ="You are a sales bot. Your primary goal is to engage the user in a conversation and persuade them to purchase a Google Pixel phone. Utilize persuasive communication, focusing on the features and benefits of the Google Pixel, to entice the user’s interest. Should the user exhibit interest, suggest they visit the shop to experience the phone firsthand and try it out. Once interest is confirmed or the user seems inclined, proceed to ask the user’s name and contact number to schedule an appointment for them to visit the shop. Be polite, professional, and responsive to the user's inquiries and concerns, addressing them appropriately to facilitate the sales process."
    
    ####sales_bot_statement = "You are a sales bot. Your main objective is to convince the user to buy a Google Pixel phone. Begin the conversation by discussing the benefits and features of Google Pixel. If the user shows interest in buying or knowing more, encourage them to visit the shop to experience the product firsthand. Be attentive to user's reactions and responses. If the user seems interested or willing to visit the shop, politely ask for their name and contact number to book an appointment for them. Say something like 'Can I have your name and contact number to schedule an appointment for you to visit our shop?'. Ensure to be courteous and maintain a friendly tone throughout the conversation, addressing any inquiries or concerns the user may have to facilitate the sales process. Give response to the last query and continue the conversation accordingly"
    #sales_bot_statement = "You are a sales bot. Your main objective is to convince the user to buy a Google Pixel phone. Begin the conversation by discussing the benefits and features of Google Pixel. Then focus on encouraging them to visit the shop to experience the product firsthand. Be attentive to user's reactions and responses. Then politely ask for their name and contact number to book an appointment for them. Say something like 'Can I have your name and contact number to schedule an appointment for you to visit our shop?'.  Give response and continue the conversation accordingly. Do not repeat the sentences that you have been already said."
    sales_bot_statement = "You are a sales bot. Your main objective is to convince the user to buy a Google Pixel phone rather than Iphone. Begin the conversation by discussing the benefits and features of Google Pixel. If the user shows interest in buying or knowing more, encourage them to visit the shop to experience the product firsthand. Be attentive to user's reactions and responses. If the user seems interested or willing to visit the shop, politely ask for their name and contact number to book an appointment for them. Say something like 'Can I have your name and contact number to schedule an appointment for you to visit our shop?'. Ensure to be courteous and maintain a friendly tone throughout the conversation, addressing any inquiries or concerns the user may have to facilitate the sales process. When they give you the name and number, end the conversation by telling then to have a great day.Give response to the last query and continue the conversation accordingly"
    
    
    while True:
        start_time = datetime.datetime.now()
        query = transcribe_stream()  # Getting the user's query.
        end_time_transcription = datetime.datetime.now()
        time_taken = (end_time_transcription - start_time).total_seconds()
        print(f'Time taken in STT: {time_taken}')

        #play_random_filler()

        if query.lower() == 'exit':
            break

        # Create structured context with separate indications for user queries and responses.
        context = ". ".join(["Query: " + entry[0] + ". Response: " + entry[1] for entry in chat_history])
        
        # Append current query to chat_history before determining the last_query.
        chat_history.append((query, ""))  # Temporary append, will update the response later.

        # Now determine the last query, which is the current query.
        last_query = '"Query": ' + ' '.join(['"' + query + '"' for _ in range(3)])

        #last_query = '"Query": "' + query + '"'


        # Emphasizing the last query by repeating it.
        emphasized_query = last_query  # As it is already repeated.

        # Combining the context with the emphasized last query to form the full_query.
        # full_query = context + ". " + emphasized_query if context else emphasized_query
        full_query = sales_bot_statement + (context + ". " + emphasized_query if context else emphasized_query)

        start_time_langchain = datetime.datetime.now()
        time_taken_before_lc = (start_time_langchain - end_time_transcription).total_seconds()
        print(f'Time taken before LangChain: {time_taken_before_lc}')
        response = chain({"question": full_query})

        end_time_langchain = datetime.datetime.now()
        time_taken_langchain = (end_time_langchain - start_time_langchain).total_seconds()
        print(f'Time taken in LangChain: {time_taken_langchain}')
        print(response)

        #Process the response from LangChain using get_similar_response
        matched_object_id, matched_response = get_similar_response(response['result'], conversation_id, query)
        end_time_faiss = datetime.datetime.now()
        time_taken_faiss = (end_time_faiss - end_time_langchain).total_seconds()
        print(f'Time taken in matching response: {time_taken_faiss}')
        print(matched_response)

        play_audio_from_id(matched_object_id)

        # # Updating the last entry in chat_history with the correct response.
        chat_history[-1] = (query, matched_response)

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
