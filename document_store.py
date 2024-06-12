from typing import Optional
from openai import OpenAI
import numpy as np
import pickle

Document = tuple[str, str]  # details, encoded_image
Embedding = list[float]


EMBEDDING_MODEL_NAME = "text-embedding-3-small"


class DocumentStore:
    def __init__(self, openai_client: OpenAI):
        self._openai_client = openai_client
        self._document_store: list[Document] = []
        self._vectors_store: list[Embedding] = []

    @property
    def documents(self) -> list[Document]:
        return self._document_store

    def add_document(self, document: Document):
        details = document[0]
        embedding = self._vectorise_text(details)
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
        response = self._openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL_NAME
        )
        return response.data[0].embedding
    
    def save_store(self, output_path: str) -> None:
        assert output_path.endswith("/")

        doc_path = output_path + "document_store.pkl"
        with open(doc_path, "wb") as document_file:
            pickle.dump(self._document_store, document_file)

        vec_path = output_path + "vectors_store.pkl"
        with open(vec_path, "wb") as vectors_file:
            pickle.dump(self._vectors_store, vectors_file)

    def load_store(self, output_path: str) -> None:
        assert output_path.endswith("/")

        doc_path = output_path + "document_store.pkl"
        with open(doc_path, "rb") as document_file:
            document_store = pickle.load(document_file)

        vec_path = output_path + "vectors_store.pkl"
        with open(vec_path, "rb") as vectors_file:
            vectors_store = pickle.load(vectors_file)

        self._document_store = document_store
        self._vectors_store = vectors_store
