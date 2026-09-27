import chromadb
from chromadb.utils import embedding_functions
import os
from typing import List, Dict, Any
from src.config import CHROMA_DB_DIR

class VectorStoreManager:
    def __init__(self, collection_name: str = "chat_messages"):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "author": c["author"],
                "channel": c["channel"],
                "timestamp": c["timestamp"]
            }
            for c in chunks
        ]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        retrieved = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0]*len(docs)
            
            for doc, meta, dist in zip(docs, metas, distances):
                retrieved.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist
                })
        return retrieved
