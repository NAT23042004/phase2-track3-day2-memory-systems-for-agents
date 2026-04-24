from abc import ABC, abstractmethod
from typing import Any, List, Dict

class BaseMemory(ABC):
    @abstractmethod
    def save(self, data: Dict[str, Any]) -> None:
        """Saves data to the memory backend."""
        pass

    @abstractmethod
    def load(self, query: str, **kwargs) -> Any:
        """Loads data from the memory backend."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears the memory store."""
        pass
