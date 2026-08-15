import json
import re
import networkx as nx
import ollama

LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'


def build_graph(json_path: str) -> nx.MultiDiGraph:
    """
    Uses MultiDiGraph, not DiGraph, because two entities can be connected by
    MORE THAN ONE relation (e.g. Ben->Carl has both "father of" and "manages").
    A plain DiGraph only allows one edge per node pair and silently overwrites
    earlier edges when a second one is added between the same pair - that was
    the bug causing "father of" to disappear when "manages" was added later.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G = nx.MultiDiGraph()
    for entry in data:
        for t in entry['triplets']:
            G.add_edge(t['subject'], t['object'], relation=t['relation'], source=entry['text'])

    return G


def get_relation_vocabulary(G: nx.MultiDiGraph):
    """Return the distinct set of relation strings actually present in the graph."""
    return sorted({data['relation'] for _, _, data in G.edges(data=True)})


def build_chain_prompt(relation_vocab: list):
    vocab_str = ', '.join(f'"{r}"' for r in relation_vocab)
    return f'''You are an information extraction assistant.
Break the user's QUESTION into an ordered chain of single-hop steps.
Each step is one (relation) to follow, starting from a known entity.
Use "?" as a placeholder for an entity that must be resolved from the graph.

CRITICAL: You may ONLY use relation words from this exact list (the graph does not contain any others):
{vocab_str}

Do not invent new relation words (e.g. never write "son" or "child of" if they are not in the list above -
if the graph only has "father of", use "father of" even for a "who is the son" question, since the same
edge answers both directions).

DIRECTION RULE (critical):
Every relation in the graph is stored as (SUBJECT, relation, OBJECT), e.g. (Ben, "father of", Carl)
means Ben IS THE FATHER, Carl is the child.
- If the question asks "X's father" / "who is the father of X" -> X is the OBJECT, we need to find the
  SUBJECT. Use "direction": "incoming" (meaning: find who points TO X with this relation).
- If the question asks "who X is the father of" / "X's child" -> X is the SUBJECT. Use "direction": "outgoing".
When in doubt for a possessive like "X's [relation]", default to "incoming" (we're looking for X's relation
FROM someone else's perspective, i.e. finding who X's [relation] IS).

Return ONLY a JSON object in this exact format:
{{"steps": [{{"subject": "...", "relation": "...", "direction": "incoming"}}, {{"subject": "?", "relation": "...", "direction": "incoming"}}]}}

The first step's "subject" must be a real entity from the question.
Every step after the first must have "subject": "?" (meaning: use the previous step's result).

Example:
Question: "Who is the father of Carl's father?"
Output: {{"steps": [{{"subject": "Carl", "relation": "father of", "direction": "incoming"}}, {{"subject": "?", "relation": "father of", "direction": "incoming"}}]}}

Example:
Question: "Where does Arthur's company's headquarters?"
Output: {{"steps": [{{"subject": "Arthur", "relation": "founded", "direction": "outgoing"}}, {{"subject": "?", "relation": "headquartered in", "direction": "outgoing"}}]}}

Example:
Question: "Who is Ben's father?"
Output: {{"steps": [{{"subject": "Ben", "relation": "father of", "direction": "incoming"}}]}}

Example:
Question: "Who is Ben the father of?"
Output: {{"steps": [{{"subject": "Ben", "relation": "father of", "direction": "outgoing"}}]}}
'''


def extract_query_chain(query: str, relation_vocab: list):
    """Ask the LLM to break the query into an ordered chain of single-hop steps,
    constrained to the graph's actual relation vocabulary, with an explicit direction per step."""
    prompt = build_chain_prompt(relation_vocab)
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query},
        ],
        format='json',
        options={'temperature': 0},
    )

    raw = response['message']['content']
    try:
        data = json.loads(raw)
        steps = data.get('steps', [])
    except json.JSONDecodeError:
        print("  [warn] Failed to parse chain JSON:", raw)
        return []

    valid_steps = []
    for s in steps:
        if isinstance(s, dict) and 'subject' in s and 'relation' in s:
            direction = s.get('direction', 'incoming').strip().lower()
            if direction not in ('incoming', 'outgoing'):
                direction = 'incoming'
            valid_steps.append({
                'subject': s['subject'].strip(),
                'relation': s['relation'].strip(),
                'direction': direction,
            })
    return valid_steps


def find_matching_node(G: nx.MultiDiGraph, entity: str):
    """Ground one extracted entity string to the closest matching node name in the graph."""
    if not entity or entity == '?':
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


def get_node_neighborhood(G: nx.MultiDiGraph, node: str):
    """
    All relations touching a single node (outgoing + incoming).
    Uses G.out_edges/G.in_edges with data=True, which correctly yields EVERY
    parallel edge between a pair (unlike G[node].items(), which only exposes
    one dict per neighbor and hides parallel edges in a MultiDiGraph).
    """
    results = []
    for _, target, data in G.out_edges(node, data=True):
        results.append((node, data['relation'], target, data.get('source')))
    for source, _, data in G.in_edges(node, data=True):
        results.append((source, data['relation'], node, data.get('source')))
    return results


def find_edge_by_relation(G: nx.MultiDiGraph, node: str, relation_hint: str, direction: str = 'incoming'):
    """
    From `node`, find an edge matching relation_hint in the SPECIFIED direction only.
      direction='incoming' -> look for edges pointing TO node (node is the object)
      direction='outgoing' -> look for edges pointing FROM node (node is the subject)
    Returns (resolved_entity, actual_relation, source_text) or None.
    """
    relation_hint_lower = relation_hint.lower()
    candidates = []

    if direction == 'outgoing':
        for _, target, data in G.out_edges(node, data=True):
            candidates.append((target, data['relation'], data.get('source')))
    else:  # incoming
        for source, _, data in G.in_edges(node, data=True):
            candidates.append((source, data['relation'], data.get('source')))

    # exact relation match first
    for entity, rel, source in candidates:
        if rel.lower() == relation_hint_lower:
            return entity, rel, source

    # fall back to substring overlap on relation text
    for entity, rel, source in candidates:
        if relation_hint_lower in rel.lower() or rel.lower() in relation_hint_lower:
            return entity, rel, source

    # last resort: try the OPPOSITE direction rather than fail outright
    opposite = 'outgoing' if direction == 'incoming' else 'incoming'
    opposite_candidates = []
    if opposite == 'outgoing':
        for _, target, data in G.out_edges(node, data=True):
            opposite_candidates.append((target, data['relation'], data.get('source')))
    else:
        for source, _, data in G.in_edges(node, data=True):
            opposite_candidates.append((source, data['relation'], data.get('source')))

    for entity, rel, source in opposite_candidates:
        if rel.lower() == relation_hint_lower:
            return entity, rel, source

    return None


def resolve_chain(G: nx.MultiDiGraph, steps: list):
    """
    Walk the chain of steps against the graph, resolving '?' from the previous step's result.
    Returns the full trace of hops taken (for showing sources) and the final answer entity.
    """
    if not steps:
        return [], None

    trace = []
    current_entity = find_matching_node(G, steps[0]['subject'])

    if not current_entity:
        print(f"  [warn] Could not ground starting entity: {steps[0]['subject']}")
        return [], None

    for step in steps:
        direction = step.get('direction', 'incoming')
        result = find_edge_by_relation(G, current_entity, step['relation'], direction)
        if not result:
            print(f"  [warn] No relation '{step['relation']}' ({direction}) found from '{current_entity}'")
            return trace, None

        resolved_entity, actual_relation, source = result
        trace.append((current_entity, actual_relation, resolved_entity, source))
        current_entity = resolved_entity

    return trace, current_entity


def retrieve_for_query(G: nx.MultiDiGraph, query: str):
    """
    Multi-hop retrieval:
      1. Extract an ordered chain of hop steps from the query, constrained to real relation vocabulary
      2. Walk the chain against the graph, resolving each '?' step by step
      3. Return the full trace of hops (for grounding/context)
    Falls back to single-entity neighborhood if chain extraction/resolution fails.
    """
    relation_vocab = get_relation_vocabulary(G)
    steps = extract_query_chain(query, relation_vocab)
    print(f"  Extracted query chain: {steps}")

    trace, final_answer = resolve_chain(G, steps)

    if trace:
        if final_answer:
            print(f"  Final answer entity: {final_answer}")
        return trace

    # fallback: crude substring match over all nodes, full neighborhood
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