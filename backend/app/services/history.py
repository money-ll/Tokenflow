from collections import deque
from datetime import datetime, timezone

class HistoryStore:
    def __init__(self, max_items=25):
        self._items = deque(maxlen=max_items)

    def add(self, result):
        item = {
            "id": result.get("id"),
            "filename": result.get("filename"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": result.get("metrics", {}),
        }
        self._items.appendleft(item)

    def items(self):
        return list(self._items)

history_store = HistoryStore()
