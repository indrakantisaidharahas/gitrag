from typing import List, Dict, Any
from src.graph_store import KnowledgeGraphManager
from src.vector_store import VectorStoreManager

class HybridRetriever:
    def __init__(self, graph_mgr: KnowledgeGraphManager, vector_mgr: VectorStoreManager):
        self.graph_mgr = graph_mgr
        self.vector_mgr = vector_mgr

    def retrieve(self, query: str, top_vector_k: int = 3) -> Dict[str, Any]:
        matched_nodes = self.graph_mgr.fuzzy_find_nodes(query)
        
        graph_triplets = []
        if matched_nodes:
            graph_triplets = self.graph_mgr.extract_subgraph(matched_nodes, max_hops=2)
            
        vector_chunks = self.vector_mgr.search(query, top_k=top_vector_k)
        
        return {
            "query": query,
            "matched_nodes": matched_nodes,
            "graph_triplets": graph_triplets,
            "vector_chunks": vector_chunks
        }
