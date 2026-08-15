import json
import networkx as nx

def build_graph(json_path: str) -> nx.DiGraph:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.DiGraph()
    for entry in data:
        for t in entry['triplets']:
            G.add_edge(t['subject'], t['object'], relation=t['relation'], source=entry['text'])

    return G


def find_matching_nodes(G: nx.DiGraph, query: str):
    """Find graph nodes whose name appears in (or overlaps with) the query text."""
    query_lower = query.lower()
    matches = []
    for node in G.nodes:
        if node.lower() in query_lower or query_lower in node.lower():
            matches.append(node)
    return matches


def get_relations_for_query(G: nx.DiGraph, query: str):
    """
    Given a user query, find matching entities in the graph and return
    all relations (outgoing + incoming) connected to them.
    """
    matched_nodes = find_matching_nodes(G, query)
    results = []

    for node in matched_nodes:
        # outgoing: node -> relation -> other
        for target, data in G[node].items():
            results.append((node, data['relation'], target, data.get('source')))

        # incoming: other -> relation -> node
        for source, _, data in G.in_edges(node, data=True):
            results.append((source, data['relation'], node, data.get('source')))

    return results


if __name__ == '__main__':
    G = build_graph('facts_triplets.json')
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    query = input('\nAsk a question: ')
    relations = get_relations_for_query(G, query)

    if not relations:
        print("No matching relations found in the graph.")
    else:
        print(f"\nFound {len(relations)} relevant relation(s):")
        for subj, rel, obj, source in relations:
            print(f"  ({subj}, {rel}, {obj})")
            print(f"    from: \"{source}\"")