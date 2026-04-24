import chromadb
import os
from typing import Any, Dict, List
from src.memory.base import BaseMemory
from chromadb.utils import embedding_functions

class SemanticMemory(BaseMemory):
    def __init__(self, collection_name="user_knowledge"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        # Use Chroma's OpenAI embedding function
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.openai_ef
        )

    def save(self, data: Dict[str, Any]) -> None:
        text = data.get("text")
        metadata = data.get("metadata", {})
        doc_id = str(hash(text))
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def load(self, query: str, **kwargs) -> List[str]:
        n_results = kwargs.get("n_results", 3)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else []

    def clear(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.openai_ef
        )
