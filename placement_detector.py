from __future__ import annotations

import re


POSITIVE_PATTERNS = [
    r"\bselected\b",
    r"\bselection\b",
    r"\bshortlisted\b",
    r"\bplaced\b",
    r"\bplacement\b",
    r"have got selected",
    r"successfully selected",
    r"final selected",
]

NEGATIVE_ONLY_PATTERNS = [
    r"registration on .* portal",
    r"campus commune portal",
]


def looks_like_selection(text: str, subject: str = "") -> bool:
    combined = f"{subject}\n{text}".lower()

    if any(re.search(p, combined) for p in POSITIVE_PATTERNS):
        return True

    return False
