import os
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
CSV_FILE_PATH = os.environ.get('CSV_FILE_PATH')

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def chat_with_user():
    # Load the documents using the CSV path from environment
    loader = CSVLoader(file_path=CSV_FILE_PATH)

    # Create an index using the loaded documents
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])

    chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.vectorstore.as_retriever(),
        input_key="question"
    )

    chat_history = []

    # Bot's opening line
    opening_line = "Hello, I'm Jacob from AryanTech Company. I'm calling you regarding our service to assist you in securing a job. Are you looking for any job opportunities right now?"
    print(opening_line)

    while True:
        query = input("Please enter your query (or type 'exit' to end): ")

        if query.lower() == 'exit':
            break

        context = ". ".join([entry[0] + ". " + entry[1] for entry in chat_history])
        full_query = context + ". " + query if context else query

        response = chain({"question": full_query})

        chat_history.append((query, response['result']))
        print(response['result'])

    return chat_history

def save_chat_to_txt(filename, chat_history):
    with open(filename, 'w') as file:
        for entry in chat_history:
            user_query, bot_response = entry
            file.write(f"User: {user_query}\nBot: {bot_response}\n\n")

if __name__ == "__main__":
    chat_history = chat_with_user()
    save_chat_to_txt('chat_history.txt', chat_history)
