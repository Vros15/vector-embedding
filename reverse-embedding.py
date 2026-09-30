#Imports necessary libraries for environment variable management, OpenAI API interaction, and mathematical operations
import ast
import math

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

#Function to calculate cosine similarity between two vectors
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    return dot_product / (magnitude_a * magnitude_b)


# Prompt the user to paste the embedding vector
embedding_string = input("Paste embedding:\n")

# Convert the pasted string into a Python list representing the embedding vector
target_embedding = ast.literal_eval(embedding_string)


# Words we want to compare against
candidates = [
    "cat",
    "dog",
    "kitten",
    "lion",
    "tiger",
    "car",
    "computer",
    "banana",
    "house",
    "fish",
]
# List to store the similarity results for each candidate word
results = []

# Iterate over each candidate word to calculate its similarity with the target embedding
for word in candidates:

    response = client.embeddings.create(
        input=word,
        model="text-embedding-3-small"
    )

    word_embedding = response.data[0].embedding

    similarity = cosine_similarity(
        target_embedding,
        word_embedding
    )

    results.append((word, similarity))


# Sort highest similarity first
results.sort(
    key=lambda x: x[1],
    reverse=True
)


print("\nClosest matches:\n")

for word, score in results:
    print(f"{word:10} {score:.4f}")


print("\nBest guess:")
print(results[0][0])