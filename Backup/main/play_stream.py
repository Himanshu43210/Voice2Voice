import os
import uuid
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  # Import ChatOpenAI
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)  # Import StreamingStdOutCallbackHandler
from stream_text_to_audio import stream_text_to_audio  # Import the function

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PLAYHT_API_KEY = os.environ.get("PLAYHT_API_KEY")
PLAYHT_USER_ID = os.environ.get("PLAYHT_USER_ID")
CSV_FILE_PATH = os.environ.get("CSV_FILE_PATH")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PLAYHT_API_KEY"] = PLAYHT_API_KEY
os.environ["PLAYHT_USER_ID"] = PLAYHT_USER_ID


def generate_unique_id():
    return str(uuid.uuid4())


conversation_id = generate_unique_id()


def chat_with_user():
    loader = CSVLoader(file_path=CSV_FILE_PATH)

    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])

    llm = ChatOpenAI(
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.vectorstore.as_retriever(),
        input_key="question",
    )

    chat_history = []

    while True:
        query = input("Enter your query (type 'exit' to end): ")

        if query.lower() == "exit":
            break

        chat_history.append((query, ""))

        last_query = '"Query": ' + " ".join(['"' + query + '"' for _ in range(3)])

        context = ". ".join(
            ["Query: " + entry[0] + ". Response: " + entry[1] for entry in chat_history]
        )
        full_query = context + ". " + last_query if context else last_query

        response = chain({"question": full_query})
        print(f"Response: {response}")
        text_response = response.get(
            "result", ""
        )  # Extract text response from the response object
        if text_response.strip():  # Check if text_response is not empty
            stream_text_to_audio(
                text_response, PLAYHT_API_KEY, PLAYHT_USER_ID
            )  # Pass API key as argumentelse:
        print("No text to convert to audio.")
        # Assuming the response contains the text to be converted to speech


if __name__ == "__main__":
    chat_history = chat_with_user()
