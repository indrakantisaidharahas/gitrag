import json
import re
import networkx as nx
import ollama

LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

QUERY_EXTRACTION_PROMPT = '''You are an information extraction assistant.
Extract entities and relation from the user's QUESTION as a triplet.
Only use words that appear in the question. Do not add outside knowledge.

Return ONLY a JSON object in this exact format:
{"subject": "...", "relation": "...", "object": "..."}

If the question only mentions one entity (no second entity), leave "object" as an empty string.
If no relation is clear, leave "relation" as an empty string.

Example:
Question: "Does Harry love Ginny?"
Output: {"subject": "Harry", "relation": "loves", "object": "Ginny"}

Example:
Question: "What do cats sleep for?"
Output: {"subject": "cats", "relation": "sleep for", "object": ""}
'''


def build_graph(json_path: str) -> nx.DiGraph:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.DiGraph()
    for entry in data:
        for t in entry['triplets']:
            G.add_edge(t['subject'], t['object'], relation=t['relation'], source=entry['text'])

    return G


def extract_query_triplet(query: str):
    """Ask the LLM to pull (subject, relation, object) out of the user's question."""
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': QUERY_EXTRACTION_PROMPT},
            {'role': 'user', 'content': query},
        ],
        format='json',
        options={'temperature': 0},
    )

    raw = response['message']['content']
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("  [warn] Failed to parse query triplet JSON:", raw)
        return {'subject': '', 'relation': '', 'object': ''}

    return {
        'subject': data.get('subject', '').strip(),
        'relation': data.get('relation', '').strip(),
        'object': data.get('object', '').strip(),
    }


def find_matching_node(G: nx.DiGraph, entity: str):
    """Ground one extracted entity string to the closest matching node name in the graph."""
    if not entity:
        return None

    entity_lower = entity.lower()

    # exact match first
    for node in G.nodes:
        if node.lower() == entity_lower:
            return node

    # fall back to substring overlap
    for node in G.nodes:
        if node.lower() in entity_lower or entity_lower in node.lower():
            return node

    return None


def get_node_neighborhood(G: nx.DiGraph, node: str):
    """1-hop relations touching a single node (outgoing + incoming)."""
    results = []
    for target, data in G[node].items():
        results.append((node, data['relation'], target, data.get('source')))
    for source, _, data in G.in_edges(node, data=True):
        results.append((source, data['relation'], node, data.get('source')))
    return results


def get_path_between(G: nx.DiGraph, source_node: str, target_node: str):
    """Find a directed path connecting two grounded entities, trying both directions."""
    for a, b in [(source_node, target_node), (target_node, source_node)]:
        try:
            path = nx.shortest_path(G, source=a, target=b)
            edges = []
            for i in range(len(path) - 1):
                data = G.get_edge_data(path[i], path[i + 1])
                edges.append((path[i], data['relation'], path[i + 1], data.get('source')))
            return edges
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return None


def retrieve_for_query(G: nx.DiGraph, query: str):
    """
    Main retrieval entry point:
      1. Extract a triplet from the query
      2. Ground subject/object to graph nodes
      3. If both grounded -> try path-finding between them
      4. Else fall back to 1-hop neighborhood of whichever entity was found
    """
    triplet = extract_query_triplet(query)
    print(f"  Extracted query triplet: {triplet}")

    subj_node = find_matching_node(G, triplet['subject'])
    obj_node = find_matching_node(G, triplet['object'])

    # Case 1: both entities grounded -> path-find between them
    if subj_node and obj_node:
        path = get_path_between(G, subj_node, obj_node)
        if path:
            return path
        # no path found -> fall back to neighborhoods of both
        return get_node_neighborhood(G, subj_node) + get_node_neighborhood(G, obj_node)

    # Case 2: only one entity grounded -> 1-hop neighborhood
    if subj_node:
        return get_node_neighborhood(G, subj_node)
    if obj_node:
        return get_node_neighborhood(G, obj_node)

    # Case 3: nothing grounded -> fall back to crude substring match over all nodes
    matches = [n for n in G.nodes if n.lower() in query.lower() or query.lower() in n.lower()]
    results = []
    for node in matches:
        results.extend(get_node_neighborhood(G, node))
    return results


if __name__ == '__main__':
    G = build_graph('facts_triplets.json')
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    query = input('\nAsk a question: ')
    relations = retrieve_for_query(G, query)

    if not relations:
        print("No matching relations found in the graph.")
    else:
        print(f"\nFound {len(relations)} relevant relation(s):")
        for subj, rel, obj, source in relations:
            print(f"  ({subj}, {rel}, {obj})")
            print(f"    from: \"{source}\"")