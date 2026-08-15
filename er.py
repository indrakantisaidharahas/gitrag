import re
import json
import os
import ollama

LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

DATASET_FILE = 'facts.txt'
OUTPUT_FILE = 'facts_triplets.json'

INSTRUCTION_PROMPT = '''You are an information extraction assistant.
Extract entities and relations from the text as triplets.

STRICT RULES:
- Only use words, numbers, and facts that literally appear in the text.
- NEVER add numbers, statistics, or facts from your own knowledge, even if they seem related or correct.
- If the text does not state a number or specific value, do not invent one.
- Each triplet must have a real subject AND a real object taken from the text. If there is no clear object, omit that triplet.
- For comparisons ("unlike X, Y does Z"), extract only the fact directly stated about the main subject. Do not make the compared-to entity the object.

SUBJECT/OBJECT DIRECTION RULE (critical):
- The SUBJECT is always the entity being described. The OBJECT is always the entity used to describe it.
- For sentences shaped "A is the [relation] of B", the sentence is describing A's relationship TO B.
  A is the subject. B is the object. Do NOT swap them.
- For sentences shaped "A is a [relation] of B" (e.g. "X is a subsidiary of Y"), A is the subject, B is the object.
- Keep the sentence's word order as your default. Only reorder if the sentence is a passive construction
  (e.g. "B was [verb]ed by A" -> subject is A, object is B).
- The relation text should read naturally left-to-right as: SUBJECT -> relation -> OBJECT.

RELATION TEXT RULE (critical):
- Keep relation phrases SHORT and CONSISTENT: 2-4 words, verb-based, no entity names inside the relation string.
- Do not put the object's name inside the relation text.
- Use a consistent phrasing for the same type of fact (e.g. always "headquartered in", never "has headquarters in"
  or "headquarters in X").

Return ONLY a JSON object in this exact format:
{"triplets": [{"subject": "...", "relation": "...", "object": "..."}]}

If no clear relation exists, return: {"triplets": []}

Example:
Text: "Cats sleep for 12 to 16 hours a day."
Output: {"triplets": [{"subject": "Cats", "relation": "sleep for", "object": "12 to 16 hours a day"}]}

Example:
Text: "Harry loves Ginny."
Output: {"triplets": [{"subject": "Harry", "relation": "loves", "object": "Ginny"}]}

Example:
Text: "Ben is the father of Carl."
Output: {"triplets": [{"subject": "Ben", "relation": "father of", "object": "Carl"}]}

Example:
Text: "Susan is the mother of Carl."
Output: {"triplets": [{"subject": "Susan", "relation": "mother of", "object": "Carl"}]}

Example:
Text: "DataSystems is a subsidiary of TechCorp."
Output: {"triplets": [{"subject": "DataSystems", "relation": "subsidiary of", "object": "TechCorp"}]}

Example:
Text: "TechCorp is headquartered in London."
Output: {"triplets": [{"subject": "TechCorp", "relation": "headquartered in", "object": "London"}]}

Example:
Text: "Arthur owns 40 percent of TechCorp."
Output: {"triplets": [{"subject": "Arthur", "relation": "owns percent of", "object": "TechCorp (40%)"}]}

Example:
Text: "Carl was born in Berlin."
Output: {"triplets": [{"subject": "Carl", "relation": "born in", "object": "Berlin"}]}
'''

# Canonical relation forms: any key phrase found in a raw relation string gets mapped
# to the canonical value. This runs AFTER extraction, so it fixes inconsistency even
# when the model produces slightly different wording across similar sentences.
RELATION_CANONICAL_MAP = {
    'father of': 'father of',
    'is the father of': 'father of',
    'mother of': 'mother of',
    'is the mother of': 'mother of',
    'married to': 'married to',
    'is married to': 'married to',
    'born in': 'born in',
    'was born in': 'born in',
    'works at': 'works at',
    'headquartered in': 'headquartered in',
    'headquarters in': 'headquartered in',
    'headquarters': 'headquartered in',
    'founded': 'founded',
    'acquired': 'acquired',
    'subsidiary of': 'subsidiary of',
    'is a subsidiary of': 'subsidiary of',
    'manages': 'manages',
    'managed by': 'managed by',
    'owns': 'owns',
    'owned by': 'owned by',
}


def split_sentences(text: str):
    """Split raw input into individual sentences, even if punctuation has no trailing space."""
    text = re.sub(r'\.(?=[A-Z])', '. ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def normalize_relation(relation: str) -> str:
    """
    Map a raw relation string to a canonical form when a known pattern matches.
    Falls back to the lowercased, stripped original if nothing matches, so unknown
    relations still pass through rather than being dropped.
    """
    rel_lower = relation.strip().lower()

    # exact match first
    if rel_lower in RELATION_CANONICAL_MAP:
        return RELATION_CANONICAL_MAP[rel_lower]

    # substring match: catch cases like "headquarters in Berlin" -> "headquartered in"
    for pattern, canonical in RELATION_CANONICAL_MAP.items():
        if pattern in rel_lower:
            return canonical

    return rel_lower


def clean_object(obj: str) -> str:
    """Strip whitespace; placeholder for further cleanup if needed later."""
    return obj.strip()


def extract_triplets(text: str):
    """Call Ollama to extract (subject, relation, object) triplets from a single sentence."""
    response = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': INSTRUCTION_PROMPT},
            {'role': 'user', 'content': text},
        ],
        format='json',
        options={'temperature': 0},
    )

    raw = response['message']['content']

    try:
        data = json.loads(raw)
        raw_triplets = data.get('triplets', [])
    except json.JSONDecodeError:
        print("  [warn] Failed to parse JSON:", raw)
        return []

    valid_triplets = []
    for t in raw_triplets:
        if not (isinstance(t, dict) and all(k in t for k in ('subject', 'relation', 'object'))):
            print(f"  [warn] Skipping malformed triplet: {t}")
            continue

        subject = t['subject'].strip()
        obj = clean_object(t['object'])
        relation = normalize_relation(t['relation'])

        if not subject or not obj:
            print(f"  [warn] Skipping triplet with empty subject/object: {t}")
            continue

        valid_triplets.append({'subject': subject, 'relation': relation, 'object': obj})

    return valid_triplets


def load_dataset(path: str):
    """Load one fact/line per entry from the dataset file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def load_existing_results(path: str):
    """Load previously saved results, so we can resume instead of redoing work."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_results(results, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def process_file(dataset_path: str, output_path: str, resume: bool = True):
    """Process every line in the dataset file, extract triplets, and save incrementally."""
    lines = load_dataset(dataset_path)

    results = load_existing_results(output_path) if resume else []
    already_done = {r['text'] for r in results}

    total = len(lines)
    for i, line in enumerate(lines, start=1):
        if line in already_done:
            continue  # skip already-processed facts when resuming

        print(f"[{i}/{total}] {line}")
        line_triplets = []

        for sentence in split_sentences(line):
            triplets = extract_triplets(sentence)
            line_triplets.extend(triplets)
            for t in triplets:
                print(f"  ({t['subject']}, {t['relation']}, {t['object']})")

        results.append({'text': line, 'triplets': line_triplets})

        # Save after every line so progress isn't lost if it crashes or is interrupted
        save_results(results, output_path)

    print(f"\nDone. {len(results)} facts processed. Saved to {output_path}")
    return results


if __name__ == '__main__':
    process_file(DATASET_FILE, OUTPUT_FILE)