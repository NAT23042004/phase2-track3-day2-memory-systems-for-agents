import redis
import json
from typing import Any, Dict, Optional
from src.memory.base import BaseMemory

class LongTermMemory(BaseMemory):
    def __init__(self, host='localhost', port=6379, db=0, user_id='default_user'):
        self.user_id = user_id
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
            self.use_redis = True
        except (redis.ConnectionError, Exception):
            print("Redis not available, falling back to in-memory dictionary.")
            self.use_redis = False
            self._local_storage = {}

    def save(self, data: Dict[str, Any]) -> None:
        key = f"user:{self.user_id}:prefs"
        if self.use_redis:
            current_prefs = self.client.get(key)
            prefs = json.loads(current_prefs) if current_prefs else {}
            prefs.update(data)
            self.client.set(key, json.dumps(prefs))
        else:
            prefs = self._local_storage.get(key, {})
            prefs.update(data)
            self._local_storage[key] = prefs

    def load(self, query: str = "", **kwargs) -> Dict[str, Any]:
        key = f"user:{self.user_id}:prefs"
        if self.use_redis:
            data = self.client.get(key)
            return json.loads(data) if data else {}
        else:
            return self._local_storage.get(key, {})

    def clear(self) -> None:
        key = f"user:{self.user_id}:prefs"
        if self.use_redis:
            self.client.delete(key)
        else:
            if key in self._local_storage:
                del self._local_storage[key]
