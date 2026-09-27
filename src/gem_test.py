# src/gem_test.py

import json
import time
import os
from typing import List

from google import genai
from google.genai import types, errors
from pydantic import BaseModel, Field
from dataset_loader import ChatDatasetLoader

# ==========================================
# CONFIG
# ==========================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )

MODEL = "gemini-3.6-flash"

client = genai.Client(api_key=API_KEY)


# ==========================================
# GEMINI OUTPUT SCHEMA
# ==========================================

class Entity(BaseModel):
    id: str = Field(
        description="Unique temporary entity ID such as e1, e2"
    )

    name: str = Field(
        description="Canonical normalized name of the entity"
    )

    type: str = Field(
        description="Entity type"
    )

    aliases: List[str] = Field(
        description="Alternative names used for this entity in the conversation"
    )


class Relationship(BaseModel):
    source: str = Field(
        description="ID of the source entity"
    )

    relationship: str = Field(
        description="Relationship type in uppercase"
    )

    target: str = Field(
        description="ID of the target entity"
    )


class KnowledgeGraph(BaseModel):
    entities: List[Entity]

    relationships: List[Relationship]


# ==========================================
# SAMPLE CHAT DATA
# ==========================================


##i have to insitanitate 
file="/home/saidharahas/gitprojects/gitrag/data/discord_chat2.json"
## how should i prroceed next
obj=ChatDatasetLoader(file)
#default 3
obj.load_messages()
chat_data=obj.format_as_chunks()



# ==========================================
# PROMPT
# ==========================================

prompt = f"""
You are an information extraction system that builds a knowledge graph
from multi-person chat conversations.

Analyze the ENTIRE conversation together before extracting information.

CONVERSATION:

{json.dumps(chat_data, indent=2)}

TASK:

Extract important entities and relationships between them.

RULES:

- Consider all messages together as one conversation.
- The author field identifies who sent each message.
- Resolve pronouns such as "I", "me", and "my" using the message author.
- Resolve references such as "it", "this", "that", and "they" using conversation context.
- Do not create pronouns as entities.
- Avoid duplicate entities.
- Normalize entity names to canonical names.
- If multiple mentions refer to the same entity, create only one entity.
- Extract only meaningful entities.
- Do not invent facts not supported by the conversation.

Allowed entity types:

Person
Organization
Project
Technology
Concept
Event
Place
Other
"""


# ==========================================
# GEMINI CALL WITH RETRY
# ==========================================

def generate_with_retry(
    client,
    model,
    prompt,
    max_retries=5
):

    for attempt in range(max_retries):

        try:

            print(
                f"\nSending request "
                f"(attempt {attempt + 1}/{max_retries})..."
            )

            response = client.models.generate_content(

                model=model,

                contents=prompt,

                config=types.GenerateContentConfig(

                    response_mime_type="application/json",

                    response_schema=KnowledgeGraph,

                    temperature=0
                )
            )

            return response


        except errors.ServerError as e:

            if attempt == max_retries - 1:
                raise

            wait_time = min(2 ** attempt, 30)

            print(
                f"Server unavailable. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


# ==========================================
# MAIN
# ==========================================

def main():

    try:

        response = generate_with_retry(

            client=client,

            model=MODEL,

            prompt=prompt
        )


        print("\n========== RAW RESPONSE ==========\n")

        print(response.text)


        print("\n========== PARSED RESPONSE ==========\n")

        # Gemini SDK automatically parses it
        result = response.parsed

        print(
            json.dumps(
                result.model_dump(),
                indent=4
            )
        )


        print("\n========== ENTITIES ==========\n")

        for entity in result.entities:

            print(
                f"{entity.id}: "
                f"{entity.name} "
                f"({entity.type})"
            )


        print("\n========== RELATIONSHIPS ==========\n")

        for relation in result.relationships:

            print(
                f"{relation.source} "
                f"--[{relation.relationship}]--> "
                f"{relation.target}"
            )


    except Exception as e:

        print("\n========== ERROR ==========\n")

        print(type(e).__name__)
        print(e)


if __name__ == "__main__":
    main()