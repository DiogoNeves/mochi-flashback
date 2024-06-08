import os
import requests
import base64
# from openai import OpenAI

MODEL_NAME = "gpt-4o"

# client = OpenAI(
#     api_key=os.environ.get("STREAM_OPEN_AI_KEY")
# )

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


image_path = "data/2024-06-08_16.27.41.png"
base64_image = encode_image(image_path)

api_key = os.environ.get("STREAM_OPEN_AI_KEY")
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}


payload = {
  "model": MODEL_NAME,
  "messages": [
    {
        "role": "system",
        "content": """You are an assistant looking through the user's macOS screen to extract relevant information about what's happening on screen and what the user is trying to achieve. This information is then going to be stored, so that we can recall later, once the user asks questions.
        When analysing a screenshot, focus on details that might be relevant to the user in the future. They may want to remember what they were doing
        in the past."""
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Provide a short description of what's happening in this screenshot."
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
  "max_tokens": 800
}


response = requests.post("https://api.openai.com/v1/chat/completions",
                         headers=headers, json=payload)

model_response = response.json()
print(model_response["choices"][0]["message"]["content"])


# response = client.chat.completions.create(
#   model=MODEL_NAME,
#   messages=[
#     {
#         "role": "system",
#         "content": """You are an assistant looking through the user's macOS screen to extract relevant information about what's happening on screen and what the user is trying to achieve. This information is then going to be stored, so that we can recall later, once the user asks questions.
# For each screenshot, describe what's happening on screen.
# do not run code for this task"""
#     },
#     {
#       "role": "user",
#       "content": [
#         {"type": "text", "text": "What’s in this image?"},
#         {
#           "type": "image_url",
#           "image_url": {
#             "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
#           },
#         },
#       ],
#     }
#   ],
#   max_tokens=300,
# )

# print(response.choices[0])