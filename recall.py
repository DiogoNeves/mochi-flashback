import os

from openai import OpenAI
from typing_extensions import TypedDict

import solara
import solara.lab

from document_store import Document, PersistentDocumentStore


class MessageDict(TypedDict):
    role: str
    content: str


API_KEY = os.environ.get("STREAM_OPEN_AI_KEY")
INFERENCE_MODEL_NAME = "gpt-4o"

STORES_FOLDER = "stores/"

ANSWER_PROMPT = ("You are an assistant looking through descriptions of"
                 " screenshots from a user to answer questions about what was"
                 " happening on screen.\n"
                 "Keep the answers concise and relevant to the user's query."
                 " A single sentence is ideal.")


openai_client = OpenAI(api_key=API_KEY)

query: solara.Reactive[str] = solara.reactive("")
answer: solara.Reactive[str] = solara.reactive("")


def create_messages(query: str) -> list[MessageDict]:
    store = PersistentDocumentStore(openai_client=openai_client,
                                    output_path=STORES_FOLDER)
    store.load_store()
    
    documents = store.search(query, top_k=3)

    return [
      {"role": "system", "content": ANSWER_PROMPT},
      {"role": "user", "content": _documents_to_prompt(documents)},
      {"role": "assistant", "content": "ok, what do you want to know?"},
      {"role": "user", "content": query}
    ]


def _documents_to_prompt(documents: list[Document]) -> str:
    prompt = ["Here's a list of screenshot descriptions:"]

    for i, (details, _) in enumerate(documents):
        prompt.append(f"{i + 1}. {details}")

    return "\n\n".join(prompt)


@solara.component
def Page():
    def send(message):
        print("Sending message")
        query.value = message

    def call_openai():
        if not query.value:
            print("No messages")
            return
        print("Calling openai")
        response = openai_client.chat.completions.create(
            model=INFERENCE_MODEL_NAME,
            messages=create_messages(query.value)  # type: ignore
        )
        answer.value = response.choices[0].message.content or ""

    task = solara.lab.use_task(call_openai, dependencies=[query.value])  # type: ignore

    with solara.Column(
        style={"width": "700px", "height": "50vh"},
    ):
        if task.pending:
            solara.Text("I'm thinking...", style={"font-size": "1rem", "padding-left": "20px"})
        if answer.value:
            solara.Text(answer.value, style={"font-size": "1rem", "padding-left": "20px"})
        solara.lab.ChatInput(send_callback=send, disabled=task.pending)