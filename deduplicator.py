from __future__ import annotations

import hashlib
import json
from typing import Any


def record_key(record: dict[str, Any]) -> str:
    important = {
        "company": record.get("company"),
        "batch": record.get("batch"),
        "role": record.get("role"),
        "selected_count": record.get("selected_count"),
        "selection_date": record.get("selection_date"),
        "selected_students": record.get("selected_students"),
    }
    blob = json.dumps(important, sort_keys=True, default=str).lower()
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []

    for record in records:
        key = record_key(record)
        if key not in seen:
            seen.add(key)
            output.append(record)

    return output
