#Import necessary libraries for environment variable management and OpenAI API interaction
#Load environment variables from the .env file

from dotenv import load_dotenv
from openai import OpenAI

#Load environment variables from the .env file
load_dotenv()

#Initialize the OpenAI client with the loaded environment variables
client = OpenAI()

#Retrieve the embedding for the input text using the OpenAI API
response = client.embeddings.create(
    input="cat",
    model="text-embedding-3-small"
)

#Extract the embedding vector from the API response
embedding = response.data[0].embedding

#Print the embedding and its details
print("Embedding:")
print(embedding)

#Print the number of dimensions in the embedding vector
print("\nDimensions:")
print(len(embedding))

#Print the token usage details from the API response
print("\nToken Usage:")
print("Prompt tokens:", response.usage.prompt_tokens)
print("Total tokens:", response.usage.total_tokens)