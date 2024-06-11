import os
import requests
import base64
from openai import OpenAI

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


def extract_details_from_screenshot(image_path: str) -> str:
  base64_image = _encode_image(image_path)
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
              "url": f"data:image/jpeg;base64,{base64_image}"
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


Embedding = list[float]

details_store: list[str] = []
vectors_store: list[Embedding] = []


def store_details(details: str) -> None:
  embedding = _vectorise_text(details)

  details_store.append(details)
  vectors_store.append(embedding)


def _vectorise_text(text: str) -> Embedding:
  response = openai_client.embeddings.create(
      input=text,
      model=EMBEDDING_MODEL_NAME
  )
  return response.data[0].embedding


def recall(query: str) -> list[str]:
  query_embedding = _vectorise_text(query)
  return []


if __name__ == "__main__":
  image_path = DATA_FOLDER + "2024-06-08_16.27.41.png"
  details = extract_details_from_screenshot(image_path)
  print(details)
  store_details(details)
