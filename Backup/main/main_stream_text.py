import os
import uuid
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  # Import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # Import StreamingStdOutCallbackHandler

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CSV_FILE_PATH = os.environ.get("CSV_FILE_PATH")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def generate_unique_id():
    return str(uuid.uuid4())


conversation_id = generate_unique_id()


def chat_with_user():
    loader = CSVLoader(file_path=CSV_FILE_PATH)

    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])

    # Create a ChatOpenAI instance for interactive chat using the OpenAI model
    llm = ChatOpenAI(
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )

    # Initialize the chain with the ChatOpenAI model and the document retriever
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.vectorstore.as_retriever(),
        input_key="question"
    )

    chat_history = []

    while True:
        query = input(
            "Enter your query (type 'exit' to end): "
        )  # Getting the user's query as text input.

        if query.lower() == "exit":
            break

        # Append current query to chat_history before determining the last_query.
        chat_history.append(
            (query, "")
        )  # Temporary append, will update the response later.

        # Now determine the last query, which is the current query.
        last_query = '"Query": ' + " ".join(['"' + query + '"' for _ in range(3)])

        # Combining the context with the emphasized last query to form the full_query.
        context = ". ".join(
            ["Query: " + entry[0] + ". Response: " + entry[1] for entry in chat_history]
        )
        full_query = context + ". " + last_query if context else last_query

        # Now, the response is streamed and handled by the callback handler
        chain({"question": full_query})


if __name__ == "__main__":
    chat_history = chat_with_user()
