"""Disk cache for geocoding and routing API responses."""

import hashlib
import json
import os
from typing import Any, Optional


class DiskCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key_path(self, namespace: str, key: str) -> str:
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        safe_key = ''.join(c if c.isalnum() else '_' for c in key[:64])
        return os.path.join(self.cache_dir, namespace, f'{safe_key}_{digest}.json')

    def get(self, namespace: str, key: str) -> Optional[Any]:
        path = self._key_path(namespace, key)
        if not os.path.exists(path):
            return None
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def set(self, namespace: str, key: str, value: Any) -> None:
        path = self._key_path(namespace, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
