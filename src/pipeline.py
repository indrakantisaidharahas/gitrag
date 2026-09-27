import json
import ollama
from typing import Dict, Any
from google import genai
from src.config import USE_GEMINI, GEMINI_API_KEY, GEMINI_MODEL, OLLAMA_CHAT_MODEL, DATA_PATH
from src.dataset_loader import ChatDatasetLoader
from src.extractor import TripletExtractor
from src.vector_store import VectorStoreManager
from src.graph_store import KnowledgeGraphManager
from src.hybrid_retriever import HybridRetriever

RAG_SYNTHESIS_PROMPT = """You are an intelligent Assistant with access to a Knowledge Graph and Raw Chat Transcripts.

Answer the user's question accurately based ONLY on the provided Context.

---
KNOWLEDGE GRAPH RELATIONSHIPS:
{graph_context}

---
RAW CHAT TRANSCRIPTS:
{vector_context}

---
USER QUESTION:
{question}

Provide a clear, direct answer. Cite relevant users, services, or ticket numbers mentioned in the context.
If the answer cannot be determined from the context, state "I could not find an answer in the chat history."
"""

class ChatGraphRAGPipeline:
    def __init__(self, data_path: str = DATA_PATH):
        self.data_loader = ChatDatasetLoader(data_path)
        self.extractor = TripletExtractor()
        self.vector_mgr = VectorStoreManager()
        self.graph_mgr = KnowledgeGraphManager()
        self.use_gemini = USE_GEMINI
        if self.use_gemini:
            self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        self.retriever = None

    def build_indices(self):
        print("Loading chat dataset...")
        chunks = self.data_loader.format_as_chunks()
        
        print("Extracting Knowledge Graph Triplets and Indexing Vector Store...")
        self.vector_mgr.add_chunks(chunks)
        
        for chunk in chunks:
            text = chunk["text"]
            triplets = self.extractor.extract(text)
            if triplets:
                print(f"Extracted {len(triplets)} triplet(s) from {chunk['id']}")
                self.graph_mgr.add_triplets(triplets, source_id=chunk["id"])
                
        print(f"Indexing complete: {len(chunks)} text chunks, {len(self.graph_mgr.get_nodes())} graph nodes, {len(self.graph_mgr.get_edges())} graph edges.")
        self.retriever = HybridRetriever(self.graph_mgr, self.vector_mgr)

    def query(self, question: str) -> Dict[str, Any]:
        if not self.retriever:
            self.build_indices()
            
        retrieval_res = self.retriever.retrieve(question)
        
        graph_lines = [
            f"({t['subject']} -> {t['relation']} -> {t['object']})"
            for t in retrieval_res["graph_triplets"]
        ]
        graph_context = "\n".join(graph_lines) if graph_lines else "No direct graph relationships found."
        
        vector_lines = [f"- {c['text']}" for c in retrieval_res["vector_chunks"]]
        vector_context = "\n".join(vector_lines) if vector_lines else "No raw chat chunks found."
        
        prompt = RAG_SYNTHESIS_PROMPT.format(
            graph_context=graph_context,
            vector_context=vector_context,
            question=question
        )
        
        answer = self._generate_answer(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "graph_triplets": retrieval_res["graph_triplets"],
            "vector_chunks": retrieval_res["vector_chunks"],
            "matched_nodes": retrieval_res["matched_nodes"]
        }

    def _generate_answer(self, prompt: str) -> str:
        if self.use_gemini:
            try:
                res = self.gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
                return res.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    print("Gemini quota reached for synthesis. Switching to local Ollama.")
                    self.use_gemini = False
                
        try:
            res = ollama.chat(
                model=OLLAMA_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            return res["message"]["content"].strip()
        except Exception as e:
            return f"Generation Error: {e}"
