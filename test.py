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

Return ONLY a JSON object in this exact format:
{"triplets": [{"subject": "...", "relation": "...", "object": "..."}]}

If no clear relation exists, return: {"triplets": []}

Example:
Text: "Cats sleep for 12 to 16 hours a day."
Output: {"triplets": [{"subject": "Cats", "relation": "sleep for", "object": "12 to 16 hours a day"}]}

Example:
Text: "Harry loves Ginny."
Output: {"triplets": [{"subject": "Harry", "relation": "loves", "object": "Ginny"}]}
'''


def split_sentences(text: str):
    """Split raw input into individual sentences, even if punctuation has no trailing space."""
    text = re.sub(r'\.(?=[A-Z])', '. ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


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
        if isinstance(t, dict) and all(k in t for k in ('subject', 'relation', 'object')):
            valid_triplets.append(t)
        else:
            print(f"  [warn] Skipping malformed triplet: {t}")
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