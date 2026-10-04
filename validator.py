from __future__ import annotations

from typing import Any


def validate_record(record: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = []

    if not record.get("company"):
        errors.append("missing company")

    count = record.get("selected_count")
    if count is not None and count < 0:
        errors.append("negative selected_count")

    for key in (
        "minimum_cgpa",
        "minimum_10th_percentage",
        "minimum_12th_percentage",
        "ctc_lpa",
        "base_salary_lpa",
        "stipend_monthly",
        "bond_duration_months",
    ):
        value = record.get(key)
        if value is not None and value < 0:
            errors.append(f"negative value: {key}")

    return (not errors, errors)
