import json
import os
from typing import List, Dict, Any


'''
chat format
  {

    "Message ID": "1543688175870808134",
    "Channel ID": "1305404472872402975",
    "Content": "Yea",
    "Attachments": "",
    "Attachment Count": "0",
    "Author ID": "1304390575402909766",
    "Author Name": "curryexpress_15338",
    "Created At": "2026-08-30T18:25:41.467000+00:00",
    "Exported At": "2026-09-02T07:45:25.290Z"
  },



'''


class ChatDatasetLoader:
    def __init__(self, file_path: str, window_size: int = 3):
        self.file_path = file_path
        self.window_size = window_size

    def load_messages(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if isinstance(data, dict):
            if "messages" in data:
                data = data["messages"]
            elif "data" in data:
                data = data["data"]
            else:
                data = [data]
                
        return data if isinstance(data, list) else []

    def format_as_chunks(self) -> List[Dict[str, Any]]:
        raw_messages = self.load_messages()
        
        parsed_messages = []
        for idx, msg in enumerate(raw_messages):
            if not isinstance(msg, dict):
                continue
                
            msg_id = self._get_val(msg, ["Message ID", "id", "message_id", "ID"], default=f"msg_{idx}")
            author = self._get_val(msg, ["Author Name", "author", "author_name", "username", "user", "Author ID"], default="Unknown")
            channel = self._get_val(msg, ["Channel Name", "Channel ID", "channel", "channel_id"], default="general")
            timestamp = self._get_val(msg, ["Created At", "timestamp", "created_at", "date"], default="")
            content = self._get_val(msg, ["Content", "content", "message", "text"], default="").strip()
            
            if content:
                parsed_messages.append({
                    # "id": msg_id,
                    "author": author,
                    # "channel": channel,
                    # "timestamp": timestamp,
                    "content": content
                })

        chunks = []
        return parsed_messages
        
        if len(parsed_messages) <= self.window_size:
            grouped_blocks = [parsed_messages]
        else:
            grouped_blocks = [
                parsed_messages[i:i + self.window_size] 
                for i in range(0, len(parsed_messages), self.window_size)
            ]

            
        # for block_idx, block in enumerate(grouped_blocks):
        #     if not block:
        #         continue
                
        #     block_id = block[0]["id"]
        #     authors = list(set(m["author"] for m in block))
        #     primary_author = authors[0] if len(authors) == 1 else ", ".join(authors)
        #     channel = block[0]["channel"]
        #     timestamp = block[0]["timestamp"]
            
        #     lines = [f"{m['author']}: {m['content']}" for m in block]
        #     combined_content = " | ".join(lines)
            
        #     formatted_text = f"[{timestamp}] In #{channel} ({primary_author}): {combined_content}"
            
        #     chunks.append({
        #         "id": block_id,
        #         "text": formatted_text,
        #         "author": primary_author,
        #         "channel": channel,
        #         "timestamp": timestamp,
        #         "content": combined_content
        #     })
            
        # return chunks

    def _get_val(self, msg: dict, keys: list, default: str = "") -> str:
        for k in keys:
            if k in msg and msg[k]:
                return str(msg[k])
        return default


