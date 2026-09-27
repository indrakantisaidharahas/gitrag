import networkx as nx
import json
import os
from typing import List, Dict, Any, Tuple
from pyvis.network import Network

class KnowledgeGraphManager:
    def __init__(self):
        self.G = nx.MultiDiGraph()

    def add_triplets(self, triplets: List[Dict[str, str]], source_id: str = ""):
        for t in triplets:
            if not isinstance(t, dict):
                continue
            subj = str(t.get("subject") or "").strip()
            rel = str(t.get("relation") or "").strip()
            obj = str(t.get("object") or "").strip()
            
            if subj and rel and obj:
                self.G.add_edge(
                    subj,
                    obj,
                    relation=rel,
                    source=source_id
                )

    def get_nodes(self) -> List[str]:
        return list(self.G.nodes())

    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        return list(self.G.edges(data=True))

    def fuzzy_find_nodes(self, query: str) -> List[str]:
        query_lower = query.lower()
        matches = []
        for node in self.G.nodes():
            node_str = str(node).lower()
            if node_str in query_lower or query_lower in node_str:
                matches.append(node)
        return matches

    def extract_subgraph(self, start_nodes: List[str], max_hops: int = 2) -> List[Dict[str, str]]:
        subgraph_edges = []
        visited_nodes = set(start_nodes)
        
        for node in start_nodes:
            if not self.G.has_node(node):
                continue
                
            for target in self.G.neighbors(node):
                edge_data = self.G.get_edge_data(node, target)
                for k, data in edge_data.items():
                    subgraph_edges.append({
                        "subject": node,
                        "relation": data.get("relation", "RELATED_TO"),
                        "object": target
                    })
                    visited_nodes.add(target)
            
            for source in self.G.predecessors(node):
                edge_data = self.G.get_edge_data(source, node)
                for k, data in edge_data.items():
                    subgraph_edges.append({
                        "subject": source,
                        "relation": data.get("relation", "RELATED_TO"),
                        "object": node
                    })
                    visited_nodes.add(source)

        unique_triplets = []
        seen = set()
        for edge in subgraph_edges:
            key = (edge["subject"], edge["relation"], edge["object"])
            if key not in seen:
                seen.add(key)
                unique_triplets.append(edge)
                
        return unique_triplets

    def export_html_visualization(self, output_path: str = "graph.html"):
        net = Network(height="600px", width="100%", notebook=False, directed=True)
        
        for node in self.G.nodes():
            net.add_node(node, label=str(node), title=str(node), color="#4A90E2", size=25)
            
        for u, v, data in self.G.edges(data=True):
            rel = data.get("relation", "")
            net.add_edge(u, v, label=rel, title=rel, color="#97C2FC")
            
        net.write_html(output_path)
        return output_path
