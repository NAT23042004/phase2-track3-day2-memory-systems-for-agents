from typing import Any, Dict, List

from langchain_classic.memory import ConversationBufferMemory

from src.memory.base import BaseMemory


class ShortTermMemory(BaseMemory):
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)

    def save(self, data: Dict[str, Any]) -> None:
        # data expected to have 'input' and 'output'
        self.memory.save_context({"input": data.get("input")}, {"output": data.get("output")})

    def load(self, query: str = "", **kwargs) -> List[Any]:
        return self.memory.load_memory_variables({}).get("history", [])

    def clear(self) -> None:
        self.memory.clear()
