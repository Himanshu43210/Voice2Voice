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
from langchain.callbacks.base import BaseCallbackHandler  
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
            'Great!': '6516ccd2baf7090789f5bb48'
        }

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
        audio_file_id = self.phrases_dict.get(combined_words)
        if audio_file_id:
            print(f"Match found in Dictionary for combined words: {combined_words}")
            await self.play_audio_async(f"{audio_file_id}.mp3")
            self.words_list.clear()
        else:
            print(f"No match found in Dictionary for combined words: {combined_words}")


    def on_llm_new_token(self, token: str, **kwargs) -> None:
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
        
        future = asyncio.run_coroutine_threadsafe(self.search_and_play_audio(combined_words), self.loop)
        future.result()


    def on_llm_end(self, *args, **kwargs) -> None:
        self.loop.stop()
        while self.loop.is_running():
            pass
        self.loop.close()
        return self.q.empty()

load_dotenv()

OPENAI_API_KEY = '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'
CSV_FILE_PATH = 'pixel.csv'

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
    chat_history = []
    
    while True:
        query = input("Enter your query (type 'exit' to end): ")
        if query.lower() == "exit":
            break
        start_time = datetime.datetime.now()
        chat_history.append((query, ""))
        last_query = '"Query": ' + " ".join(['"' + query + '"' for _ in range(3)])
        context = ". ".join(["Query: " + entry[0] + ". Response: " + entry[1] for entry in chat_history])
        full_query = context + ". " + last_query if context else last_query
        response = chain({"question": full_query})
        print(f'Response received: {response}')
        end_time = datetime.datetime.now()
        time_taken = (end_time - start_time).total_seconds()
        print(f'Time taken: {time_taken}')

if __name__ == "__main__":
    chat_history = chat_with_user()
