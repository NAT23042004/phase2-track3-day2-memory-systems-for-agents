import json
import os
from typing import Any, Dict, List

from src.memory.base import BaseMemory


class EpisodicMemory(BaseMemory):
    def __init__(self, file_path="episodes.json"):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump([], f)

    def save(self, data: Dict[str, Any]) -> None:
        # data format: {"episode": "user was confused about async", "timestamp": "..."}
        with open(self.file_path, "r+") as f:
            episodes = json.load(f)
            episodes.append(data)
            f.seek(0)
            json.dump(episodes, f, indent=2, ensure_ascii=False)
            f.truncate()

    def load(self, query: str = "", **kwargs) -> List[Dict[str, Any]]:
        with open(self.file_path, "r") as f:
            return json.load(f)

    def clear(self) -> None:
        with open(self.file_path, "w") as f:
            json.dump([], f)
