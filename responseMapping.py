from sentence_transformers import SentenceTransformer, util

# Load pretrained SBERT model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Read the 500 sentences from the file
with open("possibalResponses.txt", "r") as file:
    sentences_500 = file.readlines()

# Remove any newline characters
sentences_500 = [sentence.strip() for sentence in sentences_500]

# Compute and store embeddings for the 500 sentences
embeddings_500 = model.encode(sentences_500, convert_to_tensor=True)


# Function to find the most similar sentence
def find_most_similar(input_sentence):
    input_embedding = model.encode(input_sentence, convert_to_tensor=True)

    max_similarity = -1.0
    most_similar_sentence = None

    for i, embedding in enumerate(embeddings_500):
        similarity = util.pytorch_cos_sim(input_embedding, embedding).item()
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_sentence = sentences_500[i]

    return most_similar_sentence, max_similarity


# Testing
input_sentence = "What is your Name"
response, similarity_score = find_most_similar(input_sentence)
print(response, similarity_score)
