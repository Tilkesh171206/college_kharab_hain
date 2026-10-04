from __future__ import annotations

from typing import Any
import pandas as pd
from ml.dataset import (
    BRANCH_GROUPS,
    ROLE_CATEGORIES,
    TECH_SKILLS,
    categorize_branch,
    categorize_role,
)
from ml.models import CompanyProfile, MLSystem


def build_input_dataframe(profile: dict[str, Any]) -> pd.DataFrame:
    """
    Converts student profile dict into the exact feature DataFrame expected by ML models.
    """
    branch = profile.get("branch", "Computer Science / IT")
    degree = profile.get("degree", "B.Tech")
    role_cat = profile.get("role_category", "Software Engineering")

    skills = profile.get("skills", [])
    skills_count = len(skills)

    core_keywords = ["Python", "Java", "C++", "SQL", "Data Structures & Algorithms"]
    cloud_ai_keywords = ["Machine Learning", "Deep Learning", "Docker", "Kubernetes", "AWS", "DevOps"]

    has_core = 1 if any(k in skills for k in core_keywords) else 0
    has_cloud_ai = 1 if any(k in skills for k in cloud_ai_keywords) else 0

    row = {
        "cgpa": float(profile.get("cgpa", 7.5)),
        "tenth_pct": float(profile.get("tenth_pct", 80.0)),
        "twelfth_pct": float(profile.get("twelfth_pct", 78.0)),
        "backlogs": int(profile.get("backlogs", 0)),
        "skills_count": skills_count,
        "has_core_tech": has_core,
        "has_cloud_ai": has_cloud_ai,
        "aptitude_score": float(profile.get("aptitude_score", 75.0)),
        "coding_score": float(profile.get("coding_score", 75.0)),
        "branch_group": branch,
        "degree": degree,
        "role_category": role_cat,
    }

    return pd.DataFrame([row])


def apply_backlog_penalty(raw_ctc: float, backlogs: int) -> float:
    """
    Applies continuous, strictly monotonic penalties for active backlogs.
    Reflects the shrinking pool of eligible hiring companies in campus placement.
    """
    if backlogs <= 0:
        return round(raw_ctc, 2)

    factors = {
        1: (0.80, 5.20),
        2: (0.60, 3.80),
        3: (0.48, 3.20),
        4: (0.38, 2.70),
    }
    factor, cap = factors.get(
        backlogs,
        (max(0.20, 0.38 - 0.04 * (backlogs - 4)), max(1.80, 2.70 - 0.20 * (backlogs - 4)))
    )
    val = min(cap, raw_ctc * factor)
    floor_val = max(1.50, 2.70 - 0.20 * (backlogs - 1))
    return round(max(floor_val, min(cap, val)), 2)


def predict_placement_and_ctc(ml_sys: MLSystem, profile: dict[str, Any]) -> dict[str, Any]:
    """
    Runs classification (Tier + probabilities) and regression (CTC + range),
    enforcing realistic campus eligibility constraints and monotonic backlog penalties.
    """
    X = build_input_dataframe(profile)
    backlogs = int(profile.get("backlogs", 0))

    # 1. Base ML model predictions
    clf = ml_sys.clf_pipeline.named_steps["classifier"]
    classes = list(clf.classes_)
    raw_probs = ml_sys.clf_pipeline.predict_proba(X)[0]
    prob_dict = {classes[i]: float(raw_probs[i]) for i in range(len(classes))}

    tier1_key = next((k for k in classes if "Tier 1" in k), "Tier 1 (High >= 6.5 LPA)")
    tier2_key = next((k for k in classes if "Tier 2" in k), "Tier 2 (Mid 4.5 - 6.5 LPA)")
    tier3_key = next((k for k in classes if "Tier 3" in k), "Tier 3 (Core/Mass < 4.5 LPA)")

    pred_ctc = float(ml_sys.reg_pipeline.predict(X)[0])

    # 2. Strict Campus Placement Eligibility & Monotonic Backlog Penalties
    if backlogs == 1:
        # Tier 1 drives strictly mandate 0 active backlogs
        p_t1 = prob_dict.get(tier1_key, 0.0) * 0.05
        p_t2 = prob_dict.get(tier2_key, 0.0) * 0.65
        p_t3 = 1.0 - (p_t1 + p_t2)
        prob_dict[tier1_key] = max(0.0, p_t1)
        prob_dict[tier2_key] = max(0.0, p_t2)
        prob_dict[tier3_key] = max(0.0, p_t3)

    elif backlogs == 2:
        # Ineligible for Tier 1; severely restricted for Tier 2
        prob_dict[tier1_key] = 0.0
        p_t2 = min(prob_dict.get(tier2_key, 0.0) * 0.15, 0.08)
        prob_dict[tier2_key] = p_t2
        prob_dict[tier3_key] = 1.0 - p_t2

    elif backlogs >= 3:
        # High backlogs: restricted exclusively to Tier 3 / Core / Mass recruitment
        prob_dict[tier1_key] = 0.0
        prob_dict[tier2_key] = 0.0
        prob_dict[tier3_key] = 1.0

    # Apply continuous monotonic decrement to CTC
    pred_ctc = apply_backlog_penalty(pred_ctc, backlogs)

    # Re-normalize probabilities
    total_p = sum(prob_dict.values())
    if total_p > 0:
        prob_map = {k: round((v / total_p) * 100, 1) for k, v in prob_dict.items()}
    else:
        prob_map = {tier1_key: 0.0, tier2_key: 0.0, tier3_key: 100.0}

    # Best tier strictly matches the calibrated probabilities
    pred_tier = max(prob_map.items(), key=lambda x: x[1])[0]

    # Realistic confidence interval
    mae = ml_sys.reg_metrics["mae"]
    spread = min(0.6, mae * 0.7)
    ctc_min = max(2.0, round(pred_ctc - spread, 2))
    ctc_max = round(pred_ctc + spread, 2)

    return {
        "predicted_tier": pred_tier,
        "tier_probabilities": prob_map,
        "predicted_ctc_lpa": round(pred_ctc, 2),
        "ctc_range_lpa": (ctc_min, ctc_max),
    }


def analyze_company_fit(ml_sys: MLSystem, profile: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Evaluates student profile against historical company drives and provides:
    1. Match percentage
    2. Missing skill gaps
    3. Eligibility assessment
    """
    student_branch = profile.get("branch", "Computer Science / IT")
    student_cgpa = float(profile.get("cgpa", 7.5))
    student_skills = set(profile.get("skills", []))
    student_backlogs = int(profile.get("backlogs", 0))

    results = []

    for comp in ml_sys.company_profiles:
        # 1. Branch match
        # If any company branch keyword matches
        branch_match = False
        if any(b.lower() in student_branch.lower() for b in comp.branches) or "all" in [b.lower() for b in comp.branches]:
            branch_match = True
        elif "Computer" in student_branch and any(b in ["CS", "IT", "AI", "DS"] for b in comp.branches):
            branch_match = True

        # 2. CGPA match
        cgpa_ok = student_cgpa >= comp.min_cgpa

        # 3. Backlog check
        backlog_ok = student_backlogs == 0 if comp.avg_ctc >= 6.0 else student_backlogs <= 1

        # 4. Skill Overlap & Gap Analysis
        comp_skills = set(comp.typical_skills)
        matched_skills = student_skills.intersection(comp_skills)
        missing_skills = comp_skills.difference(student_skills)

        if comp_skills:
            skill_score = len(matched_skills) / len(comp_skills)
        else:
            skill_score = 0.6

        # Overall composite fit score (0 - 100)
        eligibility_weight = 0.40 if (branch_match and cgpa_ok and backlog_ok) else (0.20 if branch_match else 0.05)
        cgpa_factor = min(1.0, student_cgpa / 9.0) * 0.25
        skill_factor = skill_score * 0.35

        composite_fit = round((eligibility_weight + cgpa_factor + skill_factor) * 100, 1)
        composite_fit = min(98.0, composite_fit)

        if composite_fit >= 75.0 and branch_match and cgpa_ok:
            verdict = "High Fit (Strong Candidate)"
            verdict_color = "green"
        elif composite_fit >= 50.0 and branch_match:
            verdict = "Moderate Fit (Bridging Required)"
            verdict_color = "orange"
        else:
            verdict = "Low Fit (Prerequisites Needed)"
            verdict_color = "red"

        results.append({
            "company": comp.company,
            "avg_ctc": comp.avg_ctc,
            "tier": comp.tier,
            "roles": comp.roles,
            "min_cgpa": comp.min_cgpa,
            "fit_score": composite_fit,
            "verdict": verdict,
            "verdict_color": verdict_color,
            "matched_skills": sorted(list(matched_skills)),
            "missing_skills": sorted(list(missing_skills)),
            "branch_eligible": branch_match,
            "cgpa_eligible": cgpa_ok,
            "historical_hires": comp.total_selections,
        })

    results.sort(key=lambda x: x["fit_score"], reverse=True)
    return results
