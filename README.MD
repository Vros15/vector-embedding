# OpenAI Vector Embedding Demo

A small Python demo of generating text embeddings with OpenAI and comparing them by cosine similarity. It is useful for seeing how text can be represented as vectors and how vector similarity can support retrieval in a Retrieval-Augmented Generation (RAG) application.

## What the scripts do

- [`get-embedding.py`](get-embedding.py) requests an embedding for the text `cat` using OpenAI's `text-embedding-3-small` model. It prints the vector, its number of dimensions, and token usage.
- [`reverse-embedding.py`](reverse-embedding.py) asks you to paste an embedding vector, creates embeddings for a fixed list of candidate words, and ranks those candidates by cosine similarity to the pasted vector. The highest-scoring word is printed as the best guess.

Despite its name, `reverse-embedding.py` does not reverse or decode an embedding back into its original text. It performs a nearest-candidate comparison. For example, an embedding for `cat` will generally score animal-related candidates such as `kitten` or `dog` above unrelated candidates, but the scores are semantic similarity estimates, not probabilities.

## Requirements and setup

- Python 3
- An OpenAI API key

From PowerShell in this project folder, install the two packages used by the scripts:

```powershell
python -m pip install openai python-dotenv
```

Create a `.env` file in the project folder with your API key:

```text
OPENAI_API_KEY=your_api_key_here
```

The scripts call `load_dotenv()` to load this value, and the OpenAI Python client reads `OPENAI_API_KEY` from the environment. Keep the key private; `.env` is listed in `.gitignore` and should not be committed.

## Run the demo

Generate and print the embedding for `cat`:

```powershell
python .\get-embedding.py
```

Copy the printed list of numbers, then run the comparison script:

```powershell
python .\reverse-embedding.py
```

Paste the complete list at the prompt and press Enter. The script prints candidate words ordered from highest to lowest cosine similarity. Each candidate currently causes a separate API request.

## How this relates to RAG

RAG combines information retrieval with text generation. A typical flow is:

1. Split source documents into chunks and generate an embedding for each chunk.
2. Store each chunk, its embedding, and useful metadata in a vector index or database.
3. When a user asks a question, embed the question and retrieve the most similar document chunks.
4. Give the retrieved chunks and the question to a language model, which generates an answer grounded in that context.

This project demonstrates the embedding and similarity ideas behind steps 1 and 3, in a deliberately small form: it compares one input vector with embeddings for a short, hard-coded list of words. It does not currently process documents, persist vectors, retrieve document passages, or generate an answer. A fuller RAG version would replace the word list with stored document chunks and add a generation call using the retrieved context.

## Similarity calculation

`reverse-embedding.py` computes cosine similarity:

```text
similarity(a, b) = dot(a, b) / (magnitude(a) * magnitude(b))
```

For nonzero vectors, a higher score means the vectors point in a more similar direction in the embedding space. The script expects vectors with matching dimensions; it currently does not validate the pasted input or handle zero-length vectors.
