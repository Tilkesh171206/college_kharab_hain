from __future__ import annotations

from ml.dataset import build_ml_dataset, categorize_branch, categorize_role, get_ctc_tier
from ml.models import train_ml_system
from ml.predictor import analyze_company_fit, predict_placement_and_ctc


def test_dataset_generation():
    df = build_ml_dataset(seed=42)
    assert not df.empty
    assert "placement_tier" in df.columns
    assert "ctc_lpa" in df.columns
    assert "cgpa" in df.columns


def test_categorization_helpers():
    assert categorize_branch("CS (AI&DS)") == "AI & Data Science"
    assert categorize_branch("B.Tech-CSE") == "Computer Science / IT"
    assert categorize_branch("Civil") == "Civil / Mechanical"
    assert categorize_branch("MCA (CY)") == "MCA / BCA"

    assert categorize_role("Data Scientist") == "Data & AI"
    assert categorize_role("DevOps Engineer") == "Cloud & DevOps"
    assert categorize_role("Executive Recruiter") == "Business & Operations"
    assert categorize_role("Associate SDE") == "Software Engineering"

    assert get_ctc_tier(7.5) == "Tier 1 (High >= 6.5 LPA)"
    assert get_ctc_tier(5.0) == "Tier 2 (Mid 4.5 - 6.5 LPA)"
    assert get_ctc_tier(3.0) == "Tier 3 (Core/Mass < 4.5 LPA)"


def test_ml_system_training_and_inference():
    ml_sys = train_ml_system()
    assert ml_sys.clf_metrics["accuracy"] > 0.70
    assert ml_sys.reg_metrics["r2"] > 0.50

    sample_profile = {
        "branch": "Computer Science / IT",
        "degree": "B.Tech",
        "role_category": "Software Engineering",
        "cgpa": 8.0,
        "tenth_pct": 85.0,
        "twelfth_pct": 82.0,
        "backlogs": 0,
        "skills": ["Java", "Data Structures & Algorithms", "SQL"],
        "aptitude_score": 80.0,
        "coding_score": 82.0,
    }

    prediction = predict_placement_and_ctc(ml_sys, sample_profile)
    assert "predicted_tier" in prediction
    assert "predicted_ctc_lpa" in prediction
    assert prediction["predicted_ctc_lpa"] > 0

    recommendations = analyze_company_fit(ml_sys, sample_profile)
    assert len(recommendations) > 0
    first_rec = recommendations[0]
    assert "fit_score" in first_rec
    assert "missing_skills" in first_rec


def test_backlog_monotonicity():
    ml_sys = train_ml_system()
    base_profile = {
        "branch": "Computer Science / IT",
        "degree": "B.Tech",
        "role_category": "Software Engineering",
        "cgpa": 8.5,
        "tenth_pct": 88.0,
        "twelfth_pct": 85.0,
        "backlogs": 0,
        "skills": ["Java", "Data Structures & Algorithms", "SQL", "Python"],
        "aptitude_score": 85.0,
        "coding_score": 85.0,
    }

    prev_ctc = 999.0
    for b in range(5):
        pred = predict_placement_and_ctc(ml_sys, dict(base_profile, backlogs=b))
        current_ctc = pred["predicted_ctc_lpa"]
        assert current_ctc < prev_ctc, f"CTC did not decrease with more backlogs! b={b}: {current_ctc} >= {prev_ctc}"
        prev_ctc = current_ctc
