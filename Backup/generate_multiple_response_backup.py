from pymongo import MongoClient
import openai

# Initialize OpenAI's API
openai.api_key = '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'

def generate_sentences_from_response(response_text):
    initPrompt = "Create 50 sentences similar to the given sentence in different tone, grammar, etc. Separate each sentence with a _#_#_ so that I can identify. Just give the text only in the same line, no extra character or response. Only pure data: "
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=initPrompt + response_text,
        max_tokens=1000
    )
    
    # Check if the delimiter exists in the response
    if '_#_#_' not in response.choices[0]['text']:
        print(f"Delimiter not found in response for: {response_text}")
        print(response.choices[0]['text'])
        return []

    # Split the response text based on the delimiter and return the list of sentences
    return response.choices[0]['text'].split('_#_#_')

def extend_responses():
    # Connect to the MongoDB server and get the collections
    client = MongoClient('mongodb+srv://vrchatAdmin:il4FA64i1Mbeo8Ay@cluster0.r5gre5i.mongodb.net')
    db = client['salesbot']
    responses = db['responses']
    responses_ext = db['responses_ext']

    # Iterate over each document in the responses collection
    for doc in responses.find():
        original_response = doc['response']
        generated_sentences = generate_sentences_from_response(original_response)

        # Store each generated sentence in responses_ext with a reference to the original ID
        for sentence in generated_sentences:
            if sentence.strip():  # Check if the sentence is not just whitespace
                print(sentence)
                responses_ext.insert_one({
                    'text': sentence.strip(),
                    'reference_id': doc['_id']
                })

    client.close()

# Call the function
extend_responses()

