import os
import uuid
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from elevenlabs import generate, stream  # Import ElevenLabs functions

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CSV_FILE_PATH = os.environ.get("CSV_FILE_PATH")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def generate_unique_id():
    return str(uuid.uuid4())


conversation_id = generate_unique_id()


def text_stream(response):
    yield response  # Yielding the response for audio streaming


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
        input_key="question"
    )

    chat_history = []

    while True:
        query = input("Enter your query (type 'exit' to end): ")

        if query.lower() == 'exit':
            break

        chat_history.append((query, ""))
        last_query = '"Query": ' + " ".join(['"' + query + '"' for _ in range(3)])
        context = ". ".join(["Query: " + entry[0] + ". Response: " + entry[1] for entry in chat_history])
        full_query = context + ". " + last_query if context else last_query

        response = chain({"question": full_query})  # Assuming chain returns a response

        # Send the response to ElevenLabs for audio streaming
        audio_stream = generate(
            text=text_stream(response),  # Send the response to text_stream
            voice="Nicole",
            model="eleven_monolingual_v1",
            stream=True
        )
        stream(audio_stream)  # Stream the audio


if __name__ == "__main__":
    chat_history = chat_with_user()
