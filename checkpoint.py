from __future__ import annotations

from pathlib import Path


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = set()

        if self.path.exists():
            self.done = {
                line.strip()
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

    def contains(self, message_id: str) -> bool:
        return message_id in self.done

    def add(self, message_id: str):
        if not message_id:
            return
        if message_id not in self.done:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(message_id + "\n")
            self.done.add(message_id)
