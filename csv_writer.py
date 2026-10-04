from __future__ import annotations

import pandas as pd

SELECTION_COLUMNS = [
    "company",
    "selection_date",
    "batch",
    "opportunity_type",
    "role",
    "job_category",
    "selected_count",
    "selected_students",
    "eligible_degree",
    "eligible_branches",
    "eligible_batches",
    "minimum_cgpa",
    "maximum_backlogs",
    "minimum_10th_percentage",
    "minimum_12th_percentage",
    "required_skills",
    "preferred_skills",
    "ctc_lpa",
    "base_salary_lpa",
    "stipend_monthly",
    "location",
    "city",
    "state",
    "work_mode",
    "selection_process",
    "aptitude_test",
    "coding_test",
    "technical_interview",
    "hr_interview",
    "group_discussion",
    "has_bond",
    "bond_duration_months",
    "source_subject",
    "source_message_id",
]


def _serialize(value):
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return "; ".join(
                ", ".join(f"{k}={v}" for k, v in item.items() if v is not None)
                for item in value
            )
        return "; ".join(str(x) for x in value)
    return value


def write_csv(records: list[dict], path):
    rows = []
    for record in records:
        row = {column: _serialize(record.get(column)) for column in SELECTION_COLUMNS}
        rows.append(row)

    df = pd.DataFrame(rows, columns=SELECTION_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return len(df)
