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
# STORES_FOLDER = "openai_stores/"
STORES_FOLDER = "local_stores/"

INFERENCE_MODEL_NAME = "xtuner/llava-llama-3-8b-v1_1-gguf"
API_KEY = "lm-studio"
SERVER_URL = "http://localhost:1234/v1"
OPEN_AI_HEADERS = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {API_KEY}"
}

MAX_OUTPUT_TOKENS = 800
EXTRACT_PROMPT = ("You are an assistant looking through the user's macOS screen"
                  " to extract relevant information about what's happening on"
                  " screen and what the user is trying to achieve. This"
                  " information is then going to be stored, so that we can"
                  " recall later, once the user asks questions.\n"
                  "When analysing a screenshot, focus on details that might be"
                  " relevant to the user in the future. They may want to"
                  " remember what they were doing in the past.")

ANSWER_PROMPT = ("You are an assistant looking through descriptions of"
                 " screenshots from a user to answer questions about what was"
                 " happening on screen.\n"
                 "Keep the answers concise and relevant to the user's query."
                 " A single sentence is ideal.")


openai_client = OpenAI(base_url=SERVER_URL, api_key=API_KEY)


def extract_details_from_screenshot(encoded_image: str) -> str:
  user_message = ("Provide a short description of what's happening in this"
                  " screenshot.")
  completion = openai_client.chat.completions.create(
    model=INFERENCE_MODEL_NAME,
    messages=[
      {
        "role": "system",
        "content": EXTRACT_PROMPT,
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "text": user_message},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{encoded_image}"
            },
          },
        ],
      }
    ]
  )

  if not completion.choices or not completion.choices[0].message.content:
    raise ValueError("No completion choices found")

  return completion.choices[0].message.content


def _encode_image(image_path: str) -> str:
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


store = PersistentDocumentStore(output_path=STORES_FOLDER)


def _process_all_images():
  for image_file_name in os.listdir(DATA_FOLDER):
    if image_file_name == ".gitkeep":
      continue

    try:
      document = _process_image(image_file_name)
      details = document[0]
      store.add_document(details, document)
    except requests.HTTPError as e:
      print(f"Error processing {image_file_name}: {e}")


def _process_image(image_file_name: str) -> Document:
  print("======Processing======")
  print(f"Image: {image_file_name}")

  image_path = os.path.join(DATA_FOLDER, image_file_name)

  print("Encoding image")
  encoded_image = _encode_image(image_path)

  print("Extracting details")
  details = extract_details_from_screenshot(encoded_image)

  print(f"Storing: {details}")

  document = (details, encoded_image)
  store.add_document(details, document)

  return document


def recall(query: str, top_k: int) -> tuple[str, list[Document]] | None:
  documents = store.search(query, top_k)

  completion = openai_client.chat.completions.create(
    model=INFERENCE_MODEL_NAME,
    messages=[
      {"role": "system", "content": ANSWER_PROMPT},
      {"role": "user", "content": _documents_to_prompt(documents)},
      {"role": "assistant", "content": "ok, what do you want to know?"},
      {"role": "user", "content": query}
    ]
  )

  if not completion.choices[0].message.content:
    return None

  return completion.choices[0].message.content, documents


def _documents_to_prompt(documents: list[Document]) -> str:
  prompt = ["Here's a list of screenshot descriptions:"]

  for i, (details, _) in enumerate(documents):
    prompt.append(f"{i + 1}. {details}")

  return "\n\n".join(prompt)


def main() -> None:
  _process_all_images()
  store.save_store()
  print("Done processing images")


if __name__ == "__main__":
  main()
