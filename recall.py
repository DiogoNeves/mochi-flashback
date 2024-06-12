import os
import requests
import base64
from openai import OpenAI

from document_store import Document, PersistentDocumentStore

# 1. Assume screenshots exist
# 2. Extract details from the screenshot
# 3. Vectorise and store those details
# 4. Recall <- user triggered

DATA_FOLDER = "data/"
STORES_FOLDER = "stores/"

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


store = PersistentDocumentStore(openai_client=openai_client,
                                output_path=STORES_FOLDER)


def recall(query: str, top_k: int = 5) -> list[Document]:
  results = store.search(query, top_k)
  # TODO: ask llm to rank and filter the results
  return results


def _process_all_images():
  for image_file_name in os.listdir(DATA_FOLDER):
    if image_file_name == ".gitkeep":
      continue

    image_path = os.path.join(DATA_FOLDER, image_file_name)
    encoded_image = _encode_image(image_path)
    details = extract_details_from_screenshot(encoded_image)
    print("Storing: ", details)

    document = (details, encoded_image)
    store.add_document(details, document)


if __name__ == "__main__":
  store.load_store()

  if len(store.documents) == 0:
    _process_all_images()
    store.save_store()

  recall_query = "When was I using OBS?"
  results = recall(recall_query, top_k=2)

  all_details = [details for details, _ in results]
  print(f"Results: {all_details}\nCount: {len(results)}")
