import json
from typing import Any, Dict

import redis
import json
import os
from typing import Any, Dict, Optional
from src.memory.base import BaseMemory

class LongTermMemory(BaseMemory):
    def __init__(self, host='localhost', port=6379, db=0, user_id='default_user', fallback_file='user_profile.json'):
        self.user_id = user_id
        self.fallback_file = fallback_file
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True, socket_connect_timeout=1)
            self.client.ping()
            self.use_redis = True
        except (redis.ConnectionError, Exception):
            print(f"Redis not available, falling back to persistent JSON: {fallback_file}")
            self.use_redis = False
            if not os.path.exists(self.fallback_file):
                with open(self.fallback_file, 'w') as f:
                    json.dump({}, f)

    def _load_fallback(self) -> Dict[str, Any]:
        with open(self.fallback_file, 'r') as f:
            return json.load(f)

    def _save_fallback(self, data: Dict[str, Any]) -> None:
        with open(self.fallback_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save(self, data: Dict[str, Any]) -> None:
        key = f"user:{self.user_id}:prefs"
        if self.use_redis:
            current_prefs = self.client.get(key)
            prefs = json.loads(current_prefs) if current_prefs else {}
            prefs.update(data)
            self.client.set(key, json.dumps(prefs, ensure_ascii=False))
        else:
            all_data = self._load_fallback()
            prefs = all_data.get(key, {})
            prefs.update(data)
            all_data[key] = prefs
            self._save_fallback(all_data)

    def load(self, query: str = "", **kwargs) -> Dict[str, Any]:
        key = f"user:{self.user_id}:prefs"
        if self.use_redis:
            data = self.client.get(key)
            return json.loads(data) if data else {}
        else:
            all_data = self._load_fallback()
            return all_data.get(key, {})

    def clear(self) -> None:
        key = f"user:{self.user_id}:prefs"
        if self.use_redis:
            self.client.delete(key)
        else:
            all_data = self._load_fallback()
            if key in all_data:
                del all_data[key]
            self._save_fallback(all_data)

