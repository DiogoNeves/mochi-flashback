import os
import requests
import base64
import numpy as np
from openai import OpenAI
import pickle

# 1. Assume screenshots exist
# 2. Extract details from the screenshot
# 3. Vectorise and store those details
# 4. Recall <- user triggered

DATA_FOLDER = "data/"

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
INFERENCE_MODEL_NAME = "gpt-4o"
API_KEY = os.environ.get("STREAM_OPEN_AI_KEY")
OPEN_AI_HEADERS = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {API_KEY}"
}

MAX_OUTPUT_TOKENS = 800
SYSTEM_PROMPT = ("You are an assistant looking through the user's macOS screen"
                 " to extract relevant information about what's happening on"
                 " screen and what the user is trying to achieve. This"
                 " information is then going to be stored, so that we can"
                 " recall later, once the user asks questions.\n"
                 "When analysing a screenshot, focus on details that might be"
                 " relevant to the user in the future. They may want to"
                 " remember what they were doing in the past.")


openai_client = OpenAI(api_key=API_KEY)


def extract_details_from_screenshot(encoded_image: str) -> str:
  user_message = ("Provide a short description of what's happening in this"
                  " screenshot.")
  payload = {
    "model": INFERENCE_MODEL_NAME,
    "messages": [
      {
          "role": "system",
          "content": SYSTEM_PROMPT
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": user_message
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{encoded_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": MAX_OUTPUT_TOKENS
  }

  response = requests.post("https://api.openai.com/v1/chat/completions",
                           headers=OPEN_AI_HEADERS, json=payload)
  model_response = response.json()

  return model_response["choices"][0]["message"]["content"]


def _encode_image(image_path: str) -> str:
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


Document = tuple[str, str]  # details, encoded_image
Embedding = list[float]

document_store: list[Document] = []
vectors_store: list[Embedding] = []


def store_details(document: Document) -> None:
  details = document[0]
  embedding = _vectorise_text(details)
  document_store.append(document)
  vectors_store.append(embedding)


def _vectorise_text(text: str) -> Embedding:
  response = openai_client.embeddings.create(
      input=text,
      model=EMBEDDING_MODEL_NAME
  )
  return response.data[0].embedding


def search(query: str, top_k: int) -> list[Document]:
  query_embedding = _vectorise_text(query)
  query_vector = np.array(query_embedding)
  vector_db = np.array(vectors_store)

  # Compute cosine similarity
  similarity_scores = np.dot(vector_db, query_vector) / \
    (np.linalg.norm(vector_db, axis=1) * np.linalg.norm(query_vector))
  
  # Get the top_k indices with highest similarity scores
  top_indices = np.argsort(similarity_scores)[::-1][:top_k]

  # Retrieve the corresponding documents for the top results
  top_documents = [document_store[i] for i in top_indices]

  return top_documents


def recall(query: str, top_k: int = 5) -> list[Document]:
  results = search(query, top_k)
  # TODO: ask llm to rank and filter the results
  return results


def _process_all_images():
  for image_file_name in os.listdir(DATA_FOLDER):
    image_path = DATA_FOLDER + image_file_name
    encoded_image = _encode_image(image_path)
    details = extract_details_from_screenshot(encoded_image)
    print("Storing: ", details)

    document = (details, encoded_image)
    store_details(document)


def _save_stores():
  global document_store, vectors_store
  with open("document_store.pkl", "wb") as document_file:
    pickle.dump(document_store, document_file)

  with open("vectors_store.pkl", "wb") as vectors_file:
    pickle.dump(vectors_store, vectors_file)


def _load_stores():
  global document_store, vectors_store
  with open("document_store.pkl", "rb") as document_file:
    document_store = pickle.load(document_file)

  with open("vectors_store.pkl", "rb") as vectors_file:
    vectors_store = pickle.load(vectors_file)


if __name__ == "__main__":
  # _process_all_images()
  # _save_stores()

  _load_stores()
  # print("All documents", [details for details, _ in document_store])
  # print("Count: ", len(document_store))

  recall_query = "When was I using OBS?"
  results = recall(recall_query, top_k=2)
  print("Results: ", [details for details, _ in results])
