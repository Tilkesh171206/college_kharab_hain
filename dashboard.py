from __future__ import annotations

import pandas as pd
import streamlit as st

from config import SELECTION_CSV
from ml.dataset import BRANCH_GROUPS, ROLE_CATEGORIES, TECH_SKILLS
from ml.models import train_ml_system
from ml.predictor import analyze_company_fit, predict_placement_and_ctc

st.set_page_config(
    page_title="College Placement Intelligence & ML Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 College Placement Intelligence & ML Predictor")
st.caption("End-to-End Placement Intelligence: Automated TPO Data Harvesting + Applied Predictive Machine Learning")

if not SELECTION_CSV.exists():
    st.warning(f"No dataset found yet: {SELECTION_CSV}. Please run the pipeline first.")
    st.stop()

df = pd.read_csv(SELECTION_CSV)

@st.cache_resource(show_spinner="Training Machine Learning Models on Placement Data...")
def get_ml_system():
    return train_ml_system()

ml_sys = get_ml_system()

tab_analytics, tab_predictor, tab_fit, tab_eval = st.tabs([
    "📊 TPO Historical Analytics",
    "🎯 Placement Tier & CTC Predictor",
    "💡 Company Fit & Skill-Gap",
    "🔬 ML Model Metrics & Viva Defense"
])

# -------------------------------------------------------------
# TAB 1: Historical Analytics
# -------------------------------------------------------------
with tab_analytics:
    st.subheader("Historical Placement Selection Records")
    st.caption("Extracted and normalized from verified college placement office emails.")

    total_records = len(df)
    total_placed = int(df["selected_count"].fillna(1).sum()) if "selected_count" in df.columns else total_records
    unique_companies = df["company"].dropna().nunique()
    avg_ctc = pd.to_numeric(df["ctc_lpa"], errors="coerce").mean()
    max_ctc = pd.to_numeric(df["ctc_lpa"], errors="coerce").max()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hiring Drives", total_records)
    c2.metric("Total Placements", total_placed)
    c3.metric("Recruiting Companies", unique_companies)
    c4.metric("Average CTC", f"{avg_ctc:.2f} LPA" if pd.notna(avg_ctc) else "—")
    c5.metric("Highest CTC", f"{max_ctc:.1f} LPA" if pd.notna(max_ctc) else "—")

    st.markdown("---")

    col_filter1, col_filter2 = st.columns([2, 1])
    with col_filter1:
        search = st.text_input("🔍 Filter by company or role:", placeholder="e.g. Celebal, MediaAMP, Data...")
    with col_filter2:
        tier_filter = st.selectbox(
            "Filter by CTC Tier:",
            ["All Tiers", "Tier 1 (>= 6.5 LPA)", "Tier 2 (4.5 - 6.5 LPA)", "Tier 3 (< 4.5 LPA)"]
        )

    filtered_df = df.copy()
    if search:
        s_lower = search.lower()
        filtered_df = filtered_df[
            filtered_df["company"].astype(str).str.lower().str.contains(s_lower) |
            filtered_df["role"].astype(str).str.lower().str.contains(s_lower)
        ]

    if tier_filter != "All Tiers":
        ctc_series = pd.to_numeric(filtered_df["ctc_lpa"], errors="coerce")
        if "Tier 1" in tier_filter:
            filtered_df = filtered_df[ctc_series >= 6.5]
        elif "Tier 2" in tier_filter:
            filtered_df = filtered_df[(ctc_series >= 4.5) & (ctc_series < 6.5)]
        elif "Tier 3" in tier_filter:
            filtered_df = filtered_df[ctc_series < 4.5]

    st.dataframe(filtered_df, width="stretch", height=360)

    st.subheader("Hiring Volume & Package Distributions")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Top Recruiting Companies by Records**")
        top_companies = df["company"].dropna().value_counts().head(10).reset_index()
        top_companies.columns = ["Company", "Selections"]
        st.bar_chart(top_companies.set_index("Company"))

    with chart_col2:
        st.markdown("**Package (CTC LPA) Distribution**")
        ctc_clean = pd.to_numeric(df["ctc_lpa"], errors="coerce").dropna().round(1).value_counts().sort_index().reset_index()
        ctc_clean.columns = ["CTC_LPA", "Drives"]
        st.bar_chart(ctc_clean.set_index("CTC_LPA"))


# -------------------------------------------------------------
# TAB 2: Placement Tier & CTC Predictor
# -------------------------------------------------------------
with tab_predictor:
    st.subheader("Student Career Simulator & Placement Predictor")
    st.caption("Predicts likelihood of cracking Tier 1, Tier 2, or Tier 3 drives and estimates expected CTC using Random Forest models.")

    p_col1, p_col2 = st.columns([1, 1])

    with p_col1:
        st.markdown("#### 1. Academic & Branch Profile")
        branch = st.selectbox("Branch / Department", BRANCH_GROUPS, index=0)
        degree = st.selectbox("Degree", ["B.Tech", "MCA", "BCA", "M.Tech"], index=0)
        target_role = st.selectbox("Target Role Category", ROLE_CATEGORIES, index=0)
        cgpa = st.slider("Current CGPA (out of 10.0)", min_value=5.0, max_value=10.0, value=7.8, step=0.1)
        
        sub_c1, sub_c2, sub_c3 = st.columns(3)
        with sub_c1:
            tenth_pct = st.number_input("10th Mark (%)", min_value=50.0, max_value=100.0, value=85.0, step=1.0)
        with sub_c2:
            twelfth_pct = st.number_input("12th Mark (%)", min_value=50.0, max_value=100.0, value=80.0, step=1.0)
        with sub_c3:
            backlogs = st.number_input("Active Backlogs", min_value=0, max_value=6, value=0, step=1)

    with p_col2:
        st.markdown("#### 2. Technical Skillset & Test Readiness")
        default_skills = ["Python", "SQL", "Data Structures & Algorithms"]
        skills = st.multiselect("Select Acquired Skills", TECH_SKILLS, default=default_skills)

        st.markdown("**Interview & Online Test Preparation**")
        coding_score = st.slider("Coding / Problem-Solving Readiness (0 - 100)", min_value=20, max_value=100, value=75, step=5)
        aptitude_score = st.slider("Aptitude / Quantitative Readiness (0 - 100)", min_value=20, max_value=100, value=75, step=5)

    predict_btn = st.button("🚀 Predict Placement Odds & Expected CTC", type="primary", width="stretch")

    if predict_btn or "last_profile" not in st.session_state:
        profile = {
            "branch": branch,
            "degree": degree,
            "role_category": target_role,
            "cgpa": cgpa,
            "tenth_pct": tenth_pct,
            "twelfth_pct": twelfth_pct,
            "backlogs": backlogs,
            "skills": skills,
            "coding_score": coding_score,
            "aptitude_score": aptitude_score,
        }
        st.session_state["last_profile"] = profile

    active_profile = st.session_state.get("last_profile")
    if active_profile:
        prediction = predict_placement_and_ctc(ml_sys, active_profile)

        st.markdown("---")
        st.subheader("Prediction Results")

        res_c1, res_c2, res_c3 = st.columns([1.2, 1.2, 1.6])

        with res_c1:
            tier_name = prediction["predicted_tier"]
            st.markdown(f"**Predicted Placement Tier:**")
            if "Tier 1" in tier_name:
                st.success(f"### 🏆 {tier_name}")
            elif "Tier 2" in tier_name:
                st.info(f"### ⚡ {tier_name}")
            else:
                st.warning(f"### 🎯 {tier_name}")

        with res_c2:
            pred_ctc = prediction["predicted_ctc_lpa"]
            c_min, c_max = prediction["ctc_range_lpa"]
            st.markdown(f"**Estimated Starting CTC:**")
            st.metric("Expected CTC", f"{pred_ctc:.2f} LPA", f"Range: {c_min:.1f} - {c_max:.1f} LPA")

        with res_c3:
            st.markdown("**Placement Tier Probability Distribution:**")
            probs = prediction["tier_probabilities"]
            for t_label, prob_val in probs.items():
                st.write(f"**{t_label}**: `{prob_val}%`")
                st.progress(int(prob_val) / 100)

        if active_profile.get("backlogs", 0) > 0:
            b_cnt = active_profile["backlogs"]
            if b_cnt >= 2:
                st.warning(f"⚠️ **Backlog Eligibility Impact ({b_cnt} Active Backlogs):** Having 2 or more active backlogs automatically disqualifies candidates from nearly all Tier 1 (>= 6.5 LPA) and Tier 2 campus placement drives. Selection opportunities are restricted to Tier 3 / Entry level roles until cleared.")
            else:
                st.info(f"ℹ️ **Backlog Eligibility Impact (1 Active Backlog):** Almost all Tier 1 campus drives (>= 6.5 LPA) mandate 0 active backlogs at drive registration. Clearing your backlog will restore eligibility for higher salary tiers.")


# -------------------------------------------------------------
# TAB 3: Company Fit & Skill-Gap Analyzer
# -------------------------------------------------------------
with tab_fit:
    st.subheader("Company Fit & Skill Gap Analyzer")
    st.caption("Matches your candidate profile against verified hiring companies from college drives and highlights missing skills.")

    current_profile = st.session_state.get("last_profile", {
        "branch": "Computer Science / IT",
        "degree": "B.Tech",
        "role_category": "Software Engineering",
        "cgpa": 7.8,
        "tenth_pct": 85.0,
        "twelfth_pct": 80.0,
        "backlogs": 0,
        "skills": ["Python", "SQL", "Data Structures & Algorithms"],
        "coding_score": 75,
        "aptitude_score": 75,
    })

    recommendations = analyze_company_fit(ml_sys, current_profile)

    top_matches = [r for r in recommendations if r["branch_eligible"]][:6]

    for rec in top_matches:
        with st.expander(f"🏢 **{rec['company']}** — Match Score: **{rec['fit_score']}%** ({rec['verdict']})", expanded=True):
            r_c1, r_c2, r_c3 = st.columns([1, 1, 2])
            with r_c1:
                st.markdown(f"**Average CTC:** `{rec['avg_ctc']} LPA`")
                st.markdown(f"**Tier:** `{rec['tier']}`")
                st.markdown(f"**Historical Hires:** `{rec['historical_hires']} students`")
            with r_c2:
                cgpa_status = "✅ Meets Cutoff" if rec["cgpa_eligible"] else f"⚠️ Below Typical ({rec['min_cgpa']})"
                st.markdown(f"**CGPA Requirement:** {cgpa_status}")
                st.markdown(f"**Primary Roles:** {', '.join(rec['roles'][:2])}")
            with r_c3:
                st.markdown("**Skills You Have for This Company:**")
                if rec["matched_skills"]:
                    st.write(", ".join([f"`{s}`" for s in rec["matched_skills"]]))
                else:
                    st.write("*(No direct skill overlap yet)*")

                st.markdown("🎯 **Missing Skills to Bridge the Gap:**")
                if rec["missing_skills"]:
                    st.error("Missing: " + ", ".join([f"**{s}**" for s in rec["missing_skills"]]))
                else:
                    st.success("🎉 You match all typical core skills for this company!")


# -------------------------------------------------------------
# TAB 4: Model Evaluation & College Viva Defense
# -------------------------------------------------------------
with tab_eval:
    st.subheader("Machine Learning Model Performance & Academic Defense")
    st.caption("Validation metrics, feature importance, and mathematical explanations for evaluation viva.")

    m_c1, m_c2 = st.columns(2)

    with m_c1:
        st.markdown("#### 1. Placement Tier Classifier (Multi-Class Random Forest)")
        clf_metrics = ml_sys.clf_metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{clf_metrics['accuracy'] * 100:.1f}%")
        m2.metric("Precision", f"{clf_metrics['precision'] * 100:.1f}%")
        m3.metric("Recall", f"{clf_metrics['recall'] * 100:.1f}%")
        m4.metric("F1-Score", f"{clf_metrics['f1']:.3f}")

        st.markdown("**Confusion Matrix (Test Set):**")
        labels = [" ".join(l.split(" ")[:2]) for l in clf_metrics["labels"]]
        cm_df = pd.DataFrame(
            clf_metrics["confusion_matrix"],
            index=[f"Actual {l}" for l in labels],
            columns=[f"Pred {l}" for l in labels]
        )
        st.dataframe(cm_df, width="stretch")

    with m_c2:
        st.markdown("#### 2. Package (CTC) Regressor (Random Forest Regressor)")
        reg_metrics = ml_sys.reg_metrics
        r1, r2, r3 = st.columns(3)
        r1.metric("R² Score", f"{reg_metrics['r2']:.3f}")
        r2.metric("MAE", f"{reg_metrics['mae']} LPA")
        r3.metric("RMSE", f"{reg_metrics['rmse']} LPA")

        st.markdown("**Feature Importance (What drives higher placement?):**")
        top_features = pd.DataFrame(ml_sys.clf_feature_importances[:8], columns=["Feature", "Importance"])
        top_features["Importance (%)"] = (top_features["Importance"] * 100).round(1)
        st.bar_chart(top_features.set_index("Feature")["Importance (%)"])

    st.markdown("---")
    with st.expander("📚 **Frequently Asked Questions by Examiners & How to Answer Them**", expanded=False):
        st.markdown("""
        **Q1: Where did you get the data?**
        - *Answer:* The dataset was harvested live from the college Training & Placement Office (TPO) mailbox via an IMAP client. Emails were parsed (including PDF/XLSX attachments) and normalized with structured JSON schemas using Gemini LLM into a verified database of confirmed selections.

        **Q2: Why use Random Forest instead of a simple Linear / Logistic Regression?**
        - *Answer:* Placement selection and salary offers have non-linear boundaries. For instance, high coding readiness combined with a specific skill like DevOps or Cloud yields disproportionately higher salary bumps, which decision trees capture effectively without overfitting.

        **Q3: How do you prevent overfitting given the dataset size?**
        - *Answer:* We constrain tree depth (`max_depth=6`), use ensemble bagging with 120 estimators, apply standard scaling and one-hot encoding via a `ColumnTransformer` inside an isolated `scikit-learn` Pipeline, and evaluate on a held-out 20% stratified test split.

        **Q4: How does the Skill-Gap recommendation work?**
        - *Answer:* It combines historical drive eligibility constraints (branch and CGPA thresholds) with set-theoretic Jaccard overlap on verified skill profiles, computing the difference set $\\text{Skills}_{\\text{Company}} \\setminus \\text{Skills}_{\\text{Candidate}}$ to provide actionable recommendations.
        """)
