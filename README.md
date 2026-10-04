# College Placement Selection Intelligence & ML Predictor

An end-to-end intelligence system that extracts historical campus placement selections from college TPO emails and applies **predictive machine learning** to forecast student placement tiers, estimate starting packages (CTC), and analyze company-specific skill gaps.

---

## Architecture & System Overview

```text
┌────────────────────────────────────────────────────────┐
│            Phase 1: Automated TPO Data Pipeline        │
│  TPO Mailbox (IMAP SSL)                                │
│       ↓                                                │
│  MIME & Attachment Parsing (PDF, DOCX, XLSX, CSV, TXT) │
│       ↓                                                │
│  Keyword Heuristic Prefilter                           │
│       ↓                                                │
│  LLM Extraction (Gemini Structured JSON Schema)        │
│       ↓                                                │
│  Normalization, Validation, Checkpoint & Deduplication │
│       ↓                                                │
│  historical_selections.csv                             │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│         Phase 2: Applied Machine Learning Engine       │
│  [ml/dataset.py]   Feature engineering from TPO drives │
│  [ml/models.py]    Random Forest Classifier & Regressor│
│  [ml/predictor.py] Company matching & skill-gap engine │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│          Phase 3: Streamlit Interactive Dashboard      │
│  Tab 1: 📊 TPO Historical Analytics                    │
│  Tab 2: 🎯 Placement Tier & Expected CTC Predictor     │
│  Tab 3: 💡 Company Fit & Skill-Gap Recommendation      │
│  Tab 4: 🔬 Model Metrics & College Viva Defense        │
└────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Automated TPO Harvester (Data Engineering)
* **Targeted Selection Extraction**: Selects only verified final placement/recruitment announcements; ignores non-selection notices (e.g. registration deadlines, test links, webinars).
* **Multi-Format Attachment Parsing**: Ingests tables and student lists from `.pdf`, `.docx`, `.xlsx`, `.xlsm`, `.csv`, and `.txt`.
* **Zero Hallucination Guarantee**: Strict schema enforcement ensures missing criteria remain `null` rather than fabricated.
* **Fault-Tolerant Checkpointing**: Unsuccessful requests are retried on subsequent runs, while processed emails are indexed to avoid duplicate API calls.

### 2. Machine Learning Predictive Suite
* **Placement Tier & Selection Probability (Classification)**:
  * Predicts likelihood of clearing drives across three tiers:
    * **Tier 1 (High/Dream)**: CTC $\ge$ 6.5 LPA
    * **Tier 2 (Mid Tech/Premier)**: 4.5 – 6.5 LPA
    * **Tier 3 (Core/Mass)**: $<$ 4.5 LPA
  * Algorithm: Multi-class `RandomForestClassifier` with probability outputs across all tiers.
* **Expected Salary (CTC) Regressor (Regression)**:
  * Estimates starting package in LPA alongside a confidence interval based on model Mean Absolute Error (MAE).
  * Algorithm: `RandomForestRegressor`.
* **Company Fit & Skill-Gap Recommendation Engine**:
  * Evaluates candidate profiles against campus recruiters (e.g. Celebal Technologies, MediaAMP, CubexO, Auriga IT).
  * Generates a composite match score (0–100%) checking branch eligibility, CGPA cutoff, and technical skillset.
  * Explicitly highlights **missing skill gaps** needed to crack the target company.
* **Model Explainability & Viva Defense**:
  * Displays accuracy, precision, recall, macro F1-score, $R^2$, MAE, RMSE, test confusion matrix, and Gini feature importances.

---

## Project Structure

```text
college-placement-selection-intelligence/
│
├── main.py                    # CLI entry point (test-config, run)
├── config.py                  # Environment settings and storage paths
├── email_fetcher.py           # IMAP client for college mail retrieval
├── email_parser.py            # MIME multi-part message parser
├── email_cleaner.py           # HTML and whitespace sanitization
├── attachment_parser.py       # PDF, Word, Excel, and CSV parser
├── placement_detector.py      # Heuristic regex prefilter
├── prompt.py                  # Structured extraction instructions for Gemini
├── llm_extractor.py           # Google GenAI structured output client
├── models.py                  # Pydantic v2 data models & schemas
├── normalizer.py              # Data cleaning and student count normalization
├── validator.py               # Boundary and non-negativity validations
├── checkpoint.py              # Processed Message-ID tracker
├── deduplicator.py            # SHA-256 fingerprinting deduplicator
├── csv_writer.py              # Serializes datasets to UTF-8-sig CSV
├── dashboard.py               # 4-tab Streamlit dashboard
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variables template
│
├── ml/                        # Applied Machine Learning package
│   ├── __init__.py
│   ├── dataset.py             # Feature engineering & dataset builder
│   ├── models.py              # Classifier, Regressor, & evaluation metrics
│   └── predictor.py           # Inference APIs & skill gap logic
│
├── data/
│   ├── raw/                   # raw_emails.csv
│   ├── processed/             # processed_ids.txt
│   ├── extracted/             # selection_results.jsonl
│   └── historical_selections.csv
│
└── tests/
    ├── test_normalizer.py     # Tests normalization logic
    ├── test_selection_logic.py# Tests selection filtering heuristics
    └── test_ml_pipeline.py    # Tests ML training, prediction & recommendation
```

---

## Setup & Installation

### 1. Virtual Environment Setup

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file by copying `.env.example`:

```env
EMAIL_ADDRESS=your_college_email@example.com
EMAIL_APP_PASSWORD=your_gmail_app_password
PLACEMENT_OFFICE_EMAIL=tpo@yourcollege.edu
IMAP_SERVER=imap.gmail.com
IMAP_PORT=993
MAILBOX=INBOX

GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite

TEST_EMAIL_LIMIT=20
MAX_EMAIL_CHARS=50000
REQUEST_DELAY_SECONDS=1.0
MAX_RETRIES=4
```

> **Note**: For Gmail/Google Workspace accounts, generate a 16-character **App Password** from your Google Account security settings.

---

## Usage Guide

### 1. Test Configuration

Verify your IMAP credentials and Gemini API connectivity:

```powershell
python main.py test-config
```

Expected output:
```text
Email configured: True
Placement sender configured: True
Gemini configured: True
Gemini model: gemini-3.5-flash-lite
```

### 2. Fetch and Extract Selection Records

Run the pipeline to process the latest $N$ emails:

```powershell
python main.py run --limit 20
```

* Raw email content is archived in `data/raw/raw_emails.csv`.
* Structured JSONL results are appended to `data/extracted/selection_results.jsonl`.
* Normalized, deduplicated records are saved to `data/historical_selections.csv`.

### 3. Launch the Interactive Dashboard

```powershell
streamlit run dashboard.py
```

Open `http://localhost:8501` to explore:
* **Tab 1: 📊 TPO Historical Analytics**: Search historical drives, view package distributions, and filter by CTC tier.
* **Tab 2: 🎯 Placement Tier & CTC Predictor**: Input your profile (CGPA, Branch, Skills, Test Readiness) to predict tier probabilities and salary estimates.
* **Tab 3: 💡 Company Fit & Skill-Gap**: See which campus companies you qualify for and identify missing technical skills.
* **Tab 4: 🔬 ML Model Metrics & Viva Defense**: Review academic metrics (Accuracy, Macro F1, $R^2$, MAE), the confusion matrix, and feature importance rankings.

### 4. Run Unit Tests

Execute the full verification test suite:

```powershell
python -c "import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'tests'); import test_normalizer, test_selection_logic, test_ml_pipeline; test_normalizer.test_selected_count_from_students(); test_selection_logic.test_selection_email(); test_selection_logic.test_non_selection_registration_email(); test_ml_pipeline.test_dataset_generation(); test_ml_pipeline.test_categorization_helpers(); test_ml_pipeline.test_ml_system_training_and_inference(); print('All tests passed successfully!')"
```

---

## College Presentation & Viva Guide

When defending this project in an academic evaluation, explain the design using these points:

1. **Why not just build a dashboard with Gemini?**
   * *Answer*: LLMs are effective for unstructured text extraction, but cannot perform predictive modeling on unseen student profiles. Our architecture uses Gemini solely as an **Automated Data Engineering & ETL engine**, feeding clean tabular data into supervised **Machine Learning models (Random Forest)** for quantitative prediction.
2. **Why Random Forest over Linear Regression?**
   * *Answer*: Hiring outcomes and compensation tiers exhibit non-linear interactions. A student with strong coding readiness and cloud skills receives disproportionately higher packages regardless of minor CGPA differences. Decision tree ensembles naturally capture these non-linear decision boundaries without assuming Gaussian linearity.
3. **How is overfitting mitigated?**
   * *Answer*: Trees are depth-constrained (`max_depth=6`), bundled with 120 bagging estimators, trained using standard scaling and one-hot encoding pipelines, and evaluated against a held-out 20% stratified test set.
4. **How does the Skill-Gap recommendation work?**
   * *Answer*: It evaluates hard eligibility cutoffs (branches and minimum CGPA) alongside a set-theoretic Jaccard overlap on verified skill requirements, identifying the difference set $\text{Skills}_{\text{Company}} \setminus \text{Skills}_{\text{Candidate}}$ to provide clear, actionable advice.
