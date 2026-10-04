from __future__ import annotations

import re
from typing import Any

from models import SelectionRecord


def _clean_string(value: str | None) -> str | None:
    if value is None:
        return None
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def _clean_list(values: list[str]) -> list[str]:
    result = []
    for value in values:
        cleaned = _clean_string(value)
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def normalize_record(record: SelectionRecord) -> dict[str, Any]:
    data = record.model_dump()

    for key, value in list(data.items()):
        if isinstance(value, str):
            data[key] = _clean_string(value)
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            data[key] = _clean_list(value)

    if data["selected_count"] is None and data["selected_students"]:
        data["selected_count"] = len(data["selected_students"])

    return data
