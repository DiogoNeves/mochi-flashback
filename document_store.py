import os
from typing import Any, Optional
from openai import OpenAI
import numpy as np
import pickle

Document = Any
Embedding = list[float]


# API_KEY = os.getenv("STREAM_OPEN_AI_KEY")
API_KEY = "lm-studio"
SERVER_URL = "http://localhost:1234/v1"

# EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5-GGUF"

# openai_client = OpenAI(api_key=API_KEY)
openai_client = OpenAI(base_url=SERVER_URL, api_key=API_KEY)


class DocumentStore:
    def __init__(self):
        self._document_store: list[Document] = []
        self._vectors_store: list[Embedding] = []

    @property
    def documents(self) -> list[Document]:
        return self._document_store

    def add_document(self, key: str, document: Document):
        embedding = self._vectorise_text(key)
        self._document_store.append(document)
        self._vectors_store.append(embedding)
    
    def search(self, query: str, top_k: int) -> list[Document]:
        query_embedding = self._vectorise_text(query)

        query_vector = np.array(query_embedding)
        vector_db = np.array(self._vectors_store)

        # Compute cosine similarity
        similarity_scores = np.dot(vector_db, query_vector) / \
            (np.linalg.norm(vector_db, axis=1) * np.linalg.norm(query_vector))
        
        # Get the top_k indices with highest similarity scores
        top_indices = np.argsort(similarity_scores)[::-1][:top_k]

        # Retrieve the corresponding documents for the top results
        top_documents = [self._document_store[i] for i in top_indices]

        return top_documents
    
    def _vectorise_text(self, text: str) -> Embedding:
        response = openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL_NAME
        )
        return response.data[0].embedding


class PersistentDocumentStore(DocumentStore):
    """A DocumentStore that can save and load its state to/from disk.
    This should only be used for debugging and development purposes.
    It uses pickle to serialize the document and vector stores."""

    def __init__(self, output_path: Optional[str] = None):
        super().__init__()

        store_path = output_path or ""
        self._doc_path = os.path.join(store_path, "document_store.pkl")
        self._vec_path = os.path.join(store_path, "vectors_store.pkl")

    def save_store(self) -> None:
        with open(self._doc_path, "wb") as document_file:
            pickle.dump(self._document_store, document_file)
        with open(self._vec_path, "wb") as vectors_file:
            pickle.dump(self._vectors_store, vectors_file)

    def load_store(self) -> None:
        if not os.path.exists(self._doc_path) or \
            not os.path.exists(self._vec_path):
            return

        with open(self._doc_path, "rb") as document_file:
            document_store = pickle.load(document_file)

        with open(self._vec_path, "rb") as vectors_file:
            vectors_store = pickle.load(vectors_file)
        
        self._document_store = document_store
        self._vectors_store = vectors_store
