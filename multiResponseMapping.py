from sentence_transformers import SentenceTransformer, util
from datetime import datetime

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


# List of input sentences
input_sentences = [
    "Absolutely, we have numerous success stories to share! Countless clients have reached their career aspirations thanks to our team. Check out their testimonials on our site.",
    "Indeed, our portfolio includes many success stories! Through our assistance, numerous clients have realized their professional dreams. Their accounts are available for reading on our webpage.",
    "Of course, we're proud to share the success stories of our clients! With our team's guidance, many have achieved remarkable career milestones. Visit our website to delve into their journeys.",
    "We certainly do have success stories to offer! Our team has been pivotal in helping several clients attain their career objectives. Their tales are prominently featured on our online platform.",
    "Without a doubt, success stories are part of our brand! Many have climbed their career ladders with our team's support. Their narratives are chronicled on our website for your perusal.",
]

for input_sentence in input_sentences:
    start_time = datetime.now()

    response, similarity_score = find_most_similar(input_sentence)

    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()  # Time taken in seconds

    print("Input Text:", input_sentence)
    print("Output Text:", response)
    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Time Taken: {time_taken:.4f} seconds")
    print("---------------------------------------------------------")
