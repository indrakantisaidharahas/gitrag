import os
import pickle
import ollama

# File to store the pre-computed embeddings
EMBEDDINGS_FILE = 'cat_facts_embeddings.pkl'
DATASET_FILE = 'cat-facts.txt'

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

# Your existing functions
def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)


def retrieve(query, top_n=3):
  # NOTE: Ensure VECTOR_DB is not empty here
  if not VECTOR_DB:
    print("Warning: Embedding database is empty. Please run the script to populate it first.")
    return []

  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  similarities.sort(key=lambda x: x[1], reverse=True)
  return similarities[:top_n]


# Main logic: Check for existing embeddings, otherwise compute them
VECTOR_DB = []
if os.path.exists(EMBEDDINGS_FILE):
  print(f"Loading embeddings from {EMBEDDINGS_FILE}...")
  with open(EMBEDDINGS_FILE, 'rb') as f:
    VECTOR_DB = pickle.load(f)
  print("Embeddings loaded successfully.")
else:
  print("Embeddings file not found. Calculating and storing embeddings...")
  dataset = []
  with open(DATASET_FILE, 'r') as file:
    dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries from dataset')
  
  for i, chunk in enumerate(dataset):
    print(f'Adding chunk {i+1}/{len(dataset)} to the database')
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

  with open(EMBEDDINGS_FILE, 'wb') as f:
    pickle.dump(VECTOR_DB, f)
  print(f'All embeddings saved to {EMBEDDINGS_FILE}')


# RAG Retrieval and Chat logic
input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
  print(f' - (similarity: {similarity:.2f}) {chunk}')

instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''

stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)

print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)