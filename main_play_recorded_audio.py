import os
import uuid
import pymongo
import string
import subprocess
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler  # Correct import here
import datetime
from threading import Thread
from queue import Queue, Empty
from collections.abc import Generator
import sys
import sounddevice as sd
import soundfile as sf
import string

import asyncio
import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CSV_FILE_PATH = os.getenv('CSV_FILE_PATH')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class DictionaryCallback(BaseCallbackHandler):
    def __init__(self, q):
        self.q = q
        self.words_list = []
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.start_loop, args=(self.loop,))
        self.thread.start()
        
        # Initialize your dictionary here
        self.phrases_dict = {
            "I 'd be happy to help.": '65168e9dbd44fa929db5522b',
            'Do you have a specific feature in mind,': '65168e9ebd44fa929db5522c',
            'or are you open to suggestions?': '65168e9ebd44fa929db5522d',
            'Great!': '6516ccd2baf7090789f5bb48',
            'Do you have a specific feature in mind, or are you open to suggestions?': '65168e9ebd44fa929db5522d'
        }
        self.is_answer_finished = False
        self.timeout = 0.2  # Set your desired timeout value in seconds
        self.timer = None
    def start_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def play_audio_async(self, file_id):
        print("play_audio_async called")
        file_path = f"elevenlabs_audio_files_phrases/{file_id}"
        if os.path.exists(file_path):
            data, samplerate = sf.read(file_path)
            sd.play(data, samplerate)
            sd.wait()
        else:
            print(f"Error: {file_path} does not exist")

    async def search_and_play_audio(self, combined_words):
        print("search_and_play_audio called")
        if self.is_answer_finished:  # Check whether the answer is finished or not
            print("Answer is finished. Not matching.")
            return
        audio_file_id = self.phrases_dict.get(combined_words)
        if audio_file_id:
            print(f"Match found in Dictionary for combined words: {combined_words}")
            await self.play_audio_async(f"{audio_file_id}.mp3")
            self.words_list.clear()
        else:
            print(f"No match found in Dictionary for combined words: {combined_words}")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.timer:
            self.timer.cancel()
        self.is_answer_finished = False
        self.q.put(token)
        stripped_token = token.strip()
        
        if not self.words_list:
            self.words_list.append(stripped_token)
        else:
            if stripped_token in string.punctuation:
                self.words_list.append(stripped_token)
            else:
                separator = ' '
                self.words_list.append(separator + stripped_token)
        
        combined_words = ''.join(self.words_list).strip()
        
        # Check if the token is a sentence-ending punctuation
        if stripped_token in ['.', '!', '?']:
            future = asyncio.run_coroutine_threadsafe(self.search_and_play_audio(combined_words), self.loop)
            future.result()
            self.timer = self.loop.call_later(self.timeout, self.set_answer_finished)


    def set_answer_finished(self):
        print("Timer expired. Setting is_answer_finished to True.")
        self.is_answer_finished = True

    def on_llm_end(self, *args, **kwargs) -> None:
        print("on_llm_end called")  # Debug print
        self.is_answer_finished = True
        self.words_list.clear()  # Clear the words_list when on_llm_end is called
        return self.q.empty()
    
    def stop_loop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)

def generate_unique_id():
    return str(uuid.uuid4())
conversation_id = generate_unique_id()

def chat_with_user():
    loader = CSVLoader(file_path=CSV_FILE_PATH)
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])
    
    q = Queue()
    job_done = object()
    llm = ChatOpenAI(
        streaming=True,
        callbacks=[DictionaryCallback(q)],  # Use the custom callback handler here
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.vectorstore.as_retriever(),
        input_key="question"
    )
    
    opening_line = "Hello I'm Jenny from Gadget Hub. We noticed that you were looking for Google Pixel phone on our website. Are you interested in purchasing a new phone at the moment?"
    print(opening_line)
    # play_audio_from_id('6513ce13c2214c3079d135ff')
    
    chat_history = [('Response', opening_line)]
    sales_bot_statement = "You are a sales bot. Your main objective is to convince the user to buy a Google Pixel phone rather than Iphone. Begin the conversation by discussing what features they are looking for. If the user shows interest in buying or knowing more, encourage them to visit the shop to experience the product firsthand. Be attentive to user's reactions and responses. If the user seems interested or willing to visit the shop, politely ask for their name and contact number to book an appointment for them. Say something like 'Can I have your name and contact number to schedule an appointment for you to visit our shop?'. Ensure to be courteous and maintain a friendly tone throughout the conversation, addressing any inquiries or concerns the user may have to facilitate the sales process. When they give you the name and number, end the conversation by telling then to have a great day. You have been given the chat history. Give response in short to the last query only and continue the conversation accordingly. "
   
    try:
        while True:
            query = input("Enter your query (type 'exit' to end): ")
            if query.lower() == "exit":
                break
            start_time = datetime.datetime.now()
            chat_history.append((query, ""))
            last_query = '"Query": "' + query + '"'
            context = ". ".join(["Query: " + entry[0] + ". Response: " + entry[1] for entry in chat_history])
            full_query = sales_bot_statement+(context + ". " + last_query if context else last_query)
            response = chain({"question": full_query})
            print(f'Response received: {response}')
            end_time = datetime.datetime.now()
            time_taken = (end_time - start_time).total_seconds()
            print(f'Time taken: {time_taken}')
    finally:
        # Stop the event loop and join the thread when 'exit' is pressed
        llm.callbacks[0].stop_loop()
        llm.callbacks[0].thread.join()


if __name__ == "__main__":
    chat_history = chat_with_user()

