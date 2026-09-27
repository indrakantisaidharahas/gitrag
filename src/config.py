import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("key")

USE_GEMINI = False
if GEMINI_API_KEY and len(GEMINI_API_KEY.strip()) > 10:
    USE_GEMINI = True

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

OLLAMA_EMBED_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"
OLLAMA_CHAT_MODEL = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF"

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "discord_chat2.json")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
