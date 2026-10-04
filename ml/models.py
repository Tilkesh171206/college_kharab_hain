from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import SELECTION_CSV
from ml.dataset import (
    BRANCH_GROUPS,
    ROLE_CATEGORIES,
    ROLE_SKILL_MAP,
    TECH_SKILLS,
    build_ml_dataset,
    categorize_branch,
    categorize_role,
    get_ctc_tier,
)


@dataclass
class CompanyProfile:
    company: str
    avg_ctc: float
    tier: str
    roles: list[str]
    branches: list[str]
    typical_skills: list[str]
    min_cgpa: float
    total_selections: int


@dataclass
class MLSystem:
    clf_pipeline: Pipeline
    reg_pipeline: Pipeline
    feature_names: list[str]
    clf_metrics: dict[str, Any]
    reg_metrics: dict[str, Any]
    clf_feature_importances: list[tuple[str, float]]
    reg_feature_importances: list[tuple[str, float]]
    company_profiles: list[CompanyProfile]


def extract_company_profiles(csv_path: Path = SELECTION_CSV) -> list[CompanyProfile]:
    """
    Builds a knowledge base of hiring companies from historical TPO data.
    """
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    profiles = []

    grouped = df.groupby("company")
    for company, group in grouped:
        ctc_vals = pd.to_numeric(group["ctc_lpa"], errors="coerce").dropna()
        avg_ctc = float(ctc_vals.mean()) if not ctc_vals.empty else 4.5
        tier = get_ctc_tier(avg_ctc)

        roles = [str(r) for r in group["role"].dropna().unique() if str(r).strip()]
        if not roles:
            roles = ["Software Engineer / Associate"]

        branches = []
        for b_raw in group["eligible_branches"].dropna():
            for item in str(b_raw).split(";"):
                item_clean = item.strip()
                if item_clean and item_clean not in branches:
                    branches.append(item_clean)

        if not branches:
            branches = ["CS", "IT", "AI&DS", "MCA", "BCA"]

        # Aggregate typical required skills from historical roles
        skills_set = set()
        for r in roles:
            cat = categorize_role(r)
            skills_set.update(ROLE_SKILL_MAP.get(cat, []))

        # Add explicit skills if any
        for sk_raw in group["required_skills"].dropna():
            for item in str(sk_raw).split(";"):
                item_clean = item.strip()
                if item_clean:
                    skills_set.add(item_clean)

        total_selections = int(group["selected_count"].fillna(1).sum())

        # Base minimum CGPA heuristic
        if avg_ctc >= 7.0:
            min_cgpa = 7.5
        elif avg_ctc >= 5.0:
            min_cgpa = 6.5
        else:
            min_cgpa = 6.0

        profiles.append(CompanyProfile(
            company=str(company),
            avg_ctc=round(avg_ctc, 2),
            tier=tier,
            roles=roles,
            branches=branches,
            typical_skills=sorted(list(skills_set))[:8],
            min_cgpa=min_cgpa,
            total_selections=total_selections,
        ))

    profiles.sort(key=lambda p: p.avg_ctc, reverse=True)
    return profiles


def train_ml_system() -> MLSystem:
    df = build_ml_dataset()

    feature_cols = [
        "cgpa", "tenth_pct", "twelfth_pct", "backlogs",
        "skills_count", "has_core_tech", "has_cloud_ai",
        "aptitude_score", "coding_score",
        "branch_group", "degree", "role_category",
    ]

    numeric_features = [
        "cgpa", "tenth_pct", "twelfth_pct", "backlogs",
        "skills_count", "has_core_tech", "has_cloud_ai",
        "aptitude_score", "coding_score",
    ]

    categorical_features = ["branch_group", "degree", "role_category"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    X = df[feature_cols]
    y_class = df["placement_tier"]
    y_reg = df["ctc_lpa"]

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.20, random_state=42, stratify=y_class
    )

    # 1. Classification Model: Placement Tier
    clf = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    clf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    clf_pipeline.fit(X_train, y_class_train)

    y_class_pred = clf_pipeline.predict(X_test)
    labels = sorted(list(y_class.unique()))

    clf_metrics = {
        "accuracy": round(float(accuracy_score(y_class_test, y_class_pred)), 4),
        "precision": round(float(precision_score(y_class_test, y_class_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_class_test, y_class_pred, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(y_class_test, y_class_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_class_test, y_class_pred, labels=labels).tolist(),
        "labels": labels,
        "test_size": len(y_class_test),
        "train_size": len(y_class_train),
    }

    # 2. Regression Model: CTC Package (LPA)
    reg = RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42)
    reg_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", reg)])
    reg_pipeline.fit(X_train, y_reg_train)

    y_reg_pred = reg_pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_reg_test, y_reg_pred))
    rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)))
    r2 = float(r2_score(y_reg_test, y_reg_pred))

    reg_metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
    }

    # Extract transformed feature names
    cat_encoder = clf_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_names

    clf_importances = sorted(
        zip(all_feature_names, clf.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    reg_importances = sorted(
        zip(all_feature_names, reg.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )

    company_profiles = extract_company_profiles()

    return MLSystem(
        clf_pipeline=clf_pipeline,
        reg_pipeline=reg_pipeline,
        feature_names=all_feature_names,
        clf_metrics=clf_metrics,
        reg_metrics=reg_metrics,
        clf_feature_importances=clf_importances,
        reg_feature_importances=reg_importances,
        company_profiles=company_profiles,
    )
