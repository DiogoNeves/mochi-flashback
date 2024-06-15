import os
from typing import List

from openai import OpenAI
from typing_extensions import TypedDict

import solara
import solara.lab


class MessageDict(TypedDict):
    role: str
    content: str


API_KEY = os.environ.get("STREAM_OPEN_AI_KEY")
INFERENCE_MODEL_NAME = "gpt-4o"


openai_client = OpenAI(api_key=API_KEY)

messages: solara.Reactive[List[MessageDict]] = solara.reactive([])


def add_chunk_to_ai_message(chunk: str):
    messages.value = [
        *messages.value[:-1],
        {
            "role": "assistant",
            "content": messages.value[-1]["content"] + chunk,
        },
    ]


@solara.component
def Page():
    user_message_count = len([m for m in messages.value if m["role"] == "user"])

    def send(message):
        messages.value = [
            *messages.value,
            {"role": "user", "content": message},
        ]

    def call_openai():
        if user_message_count == 0:
            return

        response = openai_client.chat.completions.create(
            model=INFERENCE_MODEL_NAME,
            messages=messages.value,  # type: ignore
            stream=True,
        )
        messages.value = [*messages.value, {"role": "assistant", "content": ""}]
        for chunk in response:
            if chunk.choices[0].finish_reason == "stop":  # type: ignore
                return
            add_chunk_to_ai_message(chunk.choices[0].delta.content)  # type: ignore

    task = solara.lab.use_task(call_openai, dependencies=[user_message_count])  # type: ignore

    with solara.Column(
        style={"width": "700px", "height": "50vh"},
    ):
        with solara.lab.ChatBox():
            for item in messages.value:
                with solara.lab.ChatMessage(
                    user=item["role"] == "user",
                    avatar=False,
                    name="ChatGPT" if item["role"] == "assistant" else "User",
                    color="rgba(0,0,0, 0.06)" if item["role"] == "assistant" else "#ff991f",
                    avatar_background_color="primary" if item["role"] == "assistant" else None,
                    border_radius="20px",
                ):
                    solara.Markdown(item["content"])
        if task.pending:
            solara.Text("I'm thinking...", style={"font-size": "1rem", "padding-left": "20px"})
        solara.lab.ChatInput(send_callback=send, disabled=task.pending)