import json
import re
import ollama
from typing import List, Dict, Any
from google import genai
from google.genai import types
from src.config import USE_GEMINI, GEMINI_API_KEY, GEMINI_MODEL, OLLAMA_CHAT_MODEL

EXTRACTION_PROMPT = """You are an Open Information Extraction (OpenIE) system for building Knowledge Graphs.

Your goal is to extract ALL factual subject-relation-object triplets from the provided chat transcript.

RULES FOR OPEN RELATION EXTRACTION:
1. Do NOT restrict relations to a fixed list. The relation MUST be dynamically generated from the action or verb phrase in the text.
   Examples:
   - "Alex ate pie" -> Subject: "Alex", Relation: "ATE", Object: "pie"

2. Normalize relations to UPPERCASE with underscores (e.g. ATE, LIVES_IN, REJECTED).
3. Subject and Object must be concise entities or phrases extracted directly from the context.

JSON OUTPUT STRUCTURE:
{
  "triplets": [
    {
      "subject": "Alex",
      "relation": "ATE",
      "object": "pie"
    }
  ]
}

If no clear factual relationship exists in the text, return: {"triplets": []}
Return ONLY valid JSON.
"""

class TripletExtractor:
    def __init__(self):
        self.use_gemini = USE_GEMINI
        if self.use_gemini:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            self.ollama_model = OLLAMA_CHAT_MODEL

    def extract(self, text: str) -> List[Dict[str, str]]:
        if self.use_gemini:
            return self._extract_gemini(text)
        else:
            return self._extract_ollama(text)

    def _clean_triplets(self, raw_triplets: Any) -> List[Dict[str, str]]:
        if not isinstance(raw_triplets, list):
            return []
        cleaned = []
        for t in raw_triplets:
            if isinstance(t, dict):
                subj = str(t.get("subject") or "").strip()
                rel = str(t.get("relation") or "").strip()
                obj = str(t.get("object") or "").strip()
                if subj and rel and obj:
                    cleaned.append({"subject": subj, "relation": rel, "object": obj})
        return cleaned

    def _extract_gemini(self, text: str) -> List[Dict[str, str]]:
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[EXTRACTION_PROMPT, f"CHAT TRANSCRIPT: {text}"],
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )
            data = json.loads(response.text)
            return self._clean_triplets(data.get("triplets", []))
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print("Gemini quota/rate-limit reached. Switching instantly to local Ollama.")
                self.use_gemini = False
            return self._extract_ollama(text)

    def _extract_ollama(self, text: str) -> List[Dict[str, str]]:
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": text}
                ],
                format="json",
                options={"temperature": 0}
            )
            raw = response["message"]["content"]
            data = json.loads(raw)
            return self._clean_triplets(data.get("triplets", []))
        except Exception:
            return []
