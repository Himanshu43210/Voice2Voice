import os
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import datetime
from threading import Thread
from queue import Queue

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CSV_FILE_PATH = os.environ.get("CSV_FILE_PATH")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class DictionaryCallback(BaseCallbackHandler):
    def __init__(self, q):
        self.q = q
        self.is_answer_finished = False
        self.timeout = 0.2  # Set your desired timeout value in seconds
        self.timer = None

# <-- Add start_time as an argument
def langchain_tasks(query, full_query, chain, start_time):
    response = chain({"question": full_query + query })
    # Extract the 'result' key from the response
    answer = response.get('result', 'No answer found.')
    print(f"Answer: {answer}")  # Print the answer


def chat_with_user(userQuestion, history):
    loader = CSVLoader(file_path=CSV_FILE_PATH)
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])

    q = Queue()
    llm = ChatOpenAI(
        streaming=True,
        # Use the custom callback handler here
        # callbacks=[DictionaryCallback(q)],
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.vectorstore.as_retriever(),
        input_key="question",
    )
    initial_statement = "You are a chatbot your main task is to provide propper assistance to the user on whatever is asked by the user. "

    if userQuestion.lower() == "exit":
        return "Thank You"
    start_time = datetime.datetime.now()
    # last_query = '"Query": "' + query + '"'
    # context = ". ".join(
    #     [
    #         "Query: " + entry[0] + ". Response: " + entry[1]
    #         for entry in chat_history
    #     ]
    # )
    full_query = initial_statement + history
    # Create and start a new thread for Langchain tasks
    langchain_thread = Thread(
        target=langchain_tasks, args=(
            userQuestion, full_query, chain, start_time)
    )  # <-- Pass start_time
    langchain_thread.start()

    # Wait for the langchain_thread to finish
    langchain_thread.join()


if __name__ == "__main__":
    chat_with_user("What do  you sell and what is my name?",
                   "[ Hi, My name is Isha.]")