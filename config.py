from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    email_address: str = os.getenv("EMAIL_ADDRESS", "")
    email_app_password: str = os.getenv("EMAIL_APP_PASSWORD", "")
    placement_office_email: str = os.getenv("PLACEMENT_OFFICE_EMAIL", "")

    imap_server: str = os.getenv("IMAP_SERVER", "imap.gmail.com")
    imap_port: int = int(os.getenv("IMAP_PORT", "993"))
    mailbox: str = os.getenv("MAILBOX", "INBOX")

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    test_email_limit: int = int(os.getenv("TEST_EMAIL_LIMIT", "20"))
    max_email_chars: int = int(os.getenv("MAX_EMAIL_CHARS", "50000"))
    request_delay_seconds: float = float(os.getenv("REQUEST_DELAY_SECONDS", "1.0"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "4"))


settings = Settings()

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTRACTED_DIR = DATA_DIR / "extracted"

for directory in (RAW_DIR, PROCESSED_DIR, EXTRACTED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SELECTION_CSV = DATA_DIR / "historical_selections.csv"
CHECKPOINT_FILE = PROCESSED_DIR / "processed_ids.txt"
RESULTS_JSONL = EXTRACTED_DIR / "selection_results.jsonl"
