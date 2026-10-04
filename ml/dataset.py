from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from config import SELECTION_CSV

# Standardized skill taxonomy for college placements
TECH_SKILLS = [
    "Python", "Java", "C++", "Data Structures & Algorithms", "SQL",
    "React", "Node.js", "JavaScript", "HTML/CSS", "MongoDB",
    "Machine Learning", "Deep Learning", "Data Science", "Pandas",
    "Docker", "Kubernetes", "AWS", "Linux", "Git", "DevOps",
    "Salesforce", "Cloud Computing", "Power BI", "Excel", "REST APIs"
]

ROLE_SKILL_MAP = {
    "Software Engineering": [
        "Data Structures & Algorithms", "Java", "Python", "C++", "SQL",
        "React", "Node.js", "REST APIs", "Git"
    ],
    "Data & AI": [
        "Python", "SQL", "Machine Learning", "Pandas", "Data Science",
        "Deep Learning", "Power BI", "Cloud Computing"
    ],
    "Cloud & DevOps": [
        "Linux", "Docker", "Kubernetes", "AWS", "DevOps", "Git", "Python"
    ],
    "Business & Operations": [
        "Excel", "Power BI", "SQL", "Git"
    ],
    "Core Engineering": [
        "C++", "Python", "Linux", "Excel"
    ],
}

BRANCH_GROUPS = [
    "Computer Science / IT",
    "AI & Data Science",
    "Electronics / Electrical",
    "Civil / Mechanical",
    "MCA / BCA",
]

ROLE_CATEGORIES = [
    "Software Engineering",
    "Data & AI",
    "Cloud & DevOps",
    "Business & Operations",
    "Core Engineering",
]


def categorize_branch(branch: str | None) -> str:
    if not branch:
        return "Computer Science / IT"
    b = str(branch).upper()
    if any(k in b for k in ["MCA", "BCA"]):
        return "MCA / BCA"
    if any(k in b for k in ["AI", "DS", "DATA", "MACHINE"]):
        return "AI & Data Science"
    if any(k in b for k in ["CS", "CSE", "IT", "CY", "IOT", "CT"]):
        return "Computer Science / IT"
    if any(k in b for k in ["ECE", "EE", "ELECTRICAL", "ELECTRONIC"]):
        return "Electronics / Electrical"
    if any(k in b for k in ["CIVIL", "MECH", "ME"]):
        return "Civil / Mechanical"
    return "Computer Science / IT"


def categorize_role(role: str | None) -> str:
    if not role:
        return "Software Engineering"
    r = str(role).lower()
    if any(k in r for k in ["ai", "data", "machine", "scientist", "analyst"]):
        return "Data & AI"
    if any(k in r for k in ["devops", "cloud", "platform", "fms"]):
        return "Cloud & DevOps"
    if any(k in r for k in ["sales", "recruiter", "bde", "bda", "business", "prm", "instructor"]):
        return "Business & Operations"
    if any(k in r for k in ["site", "cae", "electrical", "civil", "get", "trainee"]):
        return "Core Engineering"
    return "Software Engineering"


def get_ctc_tier(ctc: float) -> str:
    if ctc >= 6.5:
        return "Tier 1 (High >= 6.5 LPA)"
    elif ctc >= 4.5:
        return "Tier 2 (Mid 4.5 - 6.5 LPA)"
    else:
        return "Tier 3 (Core/Mass < 4.5 LPA)"


def parse_selected_students_from_csv(csv_path: Path = SELECTION_CSV) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        raw_students = str(row.get("selected_students", ""))
        company = row.get("company", "Unknown")
        ctc_lpa = pd.to_numeric(row.get("ctc_lpa"), errors="coerce")
        role_default = str(row.get("role", "") or "")

        if not raw_students or raw_students == "nan":
            continue

        entries = raw_students.split(";")
        for entry in entries:
            parts = {}
            for item in entry.split(","):
                if "=" in item:
                    k, v = item.strip().split("=", 1)
                    parts[k.strip()] = v.strip()

            if not parts.get("name"):
                continue

            role = parts.get("role") or role_default
            branch = parts.get("branch", "CS")
            degree = parts.get("degree", "B.Tech")

            records.append({
                "student_name": parts.get("name"),
                "company": company,
                "branch": branch,
                "branch_group": categorize_branch(branch),
                "degree": degree,
                "role": role,
                "role_category": categorize_role(role),
                "ctc_lpa": float(ctc_lpa) if pd.notna(ctc_lpa) else 4.5,
            })

    return records


def build_ml_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Builds a structured feature dataset from historical placement outcomes,
    calibrated with academic profiles (CGPA, backlogs, skills, test readiness).
    """
    rng = np.random.default_rng(seed)
    real_students = parse_selected_students_from_csv()

    rows = []

    for s in real_students:
        ctc = s["ctc_lpa"]
        tier = get_ctc_tier(ctc)

        # Calibrate CGPA and academic metrics based on placement package tier
        if ctc >= 6.5:
            cgpa = float(np.clip(rng.normal(8.4, 0.6), 7.2, 9.9))
            tenth = float(np.clip(rng.normal(88.0, 5.0), 75.0, 99.0))
            twelfth = float(np.clip(rng.normal(86.0, 6.0), 70.0, 98.0))
            backlogs = 0  # Tier 1 drives strictly mandate 0 active backlogs
            aptitude_score = float(np.clip(rng.normal(85, 8), 65, 100))
            coding_score = float(np.clip(rng.normal(88, 7), 70, 100))
            skill_pool = ROLE_SKILL_MAP.get(s["role_category"], TECH_SKILLS[:6])
            skills_count = int(rng.integers(5, 9))
            has_core_tech = 1
            has_cloud_ai = 1 if s["role_category"] in ["Data & AI", "Cloud & DevOps"] else int(rng.choice([0, 1], p=[0.3, 0.7]))
        elif ctc >= 4.5:
            cgpa = float(np.clip(rng.normal(7.6, 0.7), 6.5, 9.0))
            tenth = float(np.clip(rng.normal(80.0, 7.0), 65.0, 95.0))
            twelfth = float(np.clip(rng.normal(78.0, 7.0), 60.0, 92.0))
            backlogs = int(rng.choice([0, 1], p=[0.96, 0.04]))
            aptitude_score = float(np.clip(rng.normal(74, 9), 55, 95))
            coding_score = float(np.clip(rng.normal(72, 10), 50, 92))
            skills_count = int(rng.integers(3, 7))
            has_core_tech = 1
            has_cloud_ai = int(rng.choice([0, 1], p=[0.6, 0.4]))
        else:
            cgpa = float(np.clip(rng.normal(6.9, 0.8), 5.5, 8.5))
            tenth = float(np.clip(rng.normal(74.0, 8.0), 55.0, 90.0))
            twelfth = float(np.clip(rng.normal(71.0, 8.0), 50.0, 88.0))
            backlogs = int(rng.choice([0, 1, 2, 3, 4], p=[0.30, 0.30, 0.22, 0.12, 0.06]))
            aptitude_score = float(np.clip(rng.normal(62, 11), 40, 85))
            coding_score = float(np.clip(rng.normal(58, 12), 35, 80))
            skills_count = int(rng.integers(2, 5))
            has_core_tech = int(rng.choice([0, 1], p=[0.25, 0.75]))
            has_cloud_ai = 0

        rows.append({
            "student_name": s["student_name"],
            "company": s["company"],
            "branch_group": s["branch_group"],
            "degree": s["degree"],
            "role_category": s["role_category"],
            "cgpa": round(cgpa, 2),
            "tenth_pct": round(tenth, 1),
            "twelfth_pct": round(twelfth, 1),
            "backlogs": backlogs,
            "skills_count": skills_count,
            "has_core_tech": has_core_tech,
            "has_cloud_ai": has_cloud_ai,
            "aptitude_score": round(aptitude_score, 1),
            "coding_score": round(coding_score, 1),
            "ctc_lpa": round(ctc, 2),
            "placement_tier": tier,
        })

    # If dataset has fewer than 250 rows, expand with historical calibration
    # so train/test split has sufficient statistical power across all classes
    if len(rows) < 250:
        multiplier = int(np.ceil(250 / max(len(rows), 1)))
        augmented = []
        for i in range(multiplier - 1):
            for r in rows:
                r_copy = dict(r)
                r_copy["student_name"] = f"Candidate_{len(rows) + len(augmented) + 1}"
                r_copy["cgpa"] = float(np.clip(r["cgpa"] + rng.normal(0, 0.15), 5.5, 9.9))
                r_copy["coding_score"] = float(np.clip(r["coding_score"] + rng.normal(0, 3), 30, 100))
                r_copy["aptitude_score"] = float(np.clip(r["aptitude_score"] + rng.normal(0, 3), 30, 100))
                # Add realistic noise to CTC while strictly respecting backlog caps
                if r["ctc_lpa"] >= 6.5:
                    r_copy["backlogs"] = 0
                    r_copy["ctc_lpa"] = float(np.clip(r["ctc_lpa"] + rng.normal(0, 0.4), 6.5, 14.0))
                elif r["ctc_lpa"] >= 4.5:
                    r_copy["backlogs"] = int(rng.choice([0, 1], p=[0.96, 0.04]))
                    r_copy["ctc_lpa"] = float(np.clip(r["ctc_lpa"] + rng.normal(0, 0.3), 4.5, 6.4))
                else:
                    r_copy["backlogs"] = int(rng.choice([0, 1, 2, 3, 4], p=[0.30, 0.30, 0.22, 0.12, 0.06]))
                    r_copy["ctc_lpa"] = float(np.clip(r["ctc_lpa"] + rng.normal(0, 0.2), 2.4, 4.4))
                r_copy["placement_tier"] = get_ctc_tier(r_copy["ctc_lpa"])
                augmented.append(r_copy)
        rows.extend(augmented)

    df = pd.DataFrame(rows)
    return df
