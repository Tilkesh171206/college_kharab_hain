from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import (
    CHECKPOINT_FILE,
    EXTRACTED_DIR,
    RESULTS_JSONL,
    RAW_DIR,
    SELECTION_CSV,
    settings,
)
from csv_writer import write_csv
from deduplicator import deduplicate
from email_fetcher import IMAPFetcher
from llm_extractor import GeminiExtractor
from normalizer import normalize_record
from placement_detector import looks_like_selection
from validator import validate_record
from checkpoint import Checkpoint


def save_raw(emails):
    path = RAW_DIR / "raw_emails.csv"
    pd.DataFrame(emails).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_existing_records() -> list[dict]:
    if not RESULTS_JSONL.exists():
        return []

    records = []
    for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            records.extend(obj.get("records", []))
        except json.JSONDecodeError:
            continue
    return records


def append_result(result: dict):
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def run(limit: int):
    fetcher = IMAPFetcher()
    extractor = GeminiExtractor()
    checkpoint = Checkpoint(CHECKPOINT_FILE)

    try:
        emails = fetcher.fetch(limit=limit)
    finally:
        fetcher.close()

    raw_path = save_raw(emails)
    print(f"Fetched {len(emails)} emails -> {raw_path}")

    all_records = load_existing_records()

    for index, item in enumerate(emails, start=1):
        message_id = item.get("message_id") or item.get("imap_uid") or f"uid:{index}"

        if checkpoint.contains(message_id):
            print(f"Skipping already processed {index}/{len(emails)}")
            continue

        subject = item.get("subject", "")
        body = item.get("body", "")

        # Cheap prefilter: obvious selection language goes to Gemini.
        # We do NOT use this as the final decision because wording varies.
        candidate = looks_like_selection(body, subject)

        if not candidate:
            print(f"Processed {index}/{len(emails)} -> ignored by selection prefilter")
            checkpoint.add(message_id)
            append_result({
                "message_id": message_id,
                "subject": subject,
                "is_selection_email": False,
                "records": [],
                "prefilter": False,
            })
            continue

        try:
            result = extractor.extract(item)

            valid_records = []
            for record in result.records:
                normalized = normalize_record(record)
                normalized["source_subject"] = subject
                normalized["source_message_id"] = message_id

                ok, errors = validate_record(normalized)
                if ok:
                    valid_records.append(normalized)
                else:
                    print(f"  Invalid record skipped: {errors}")

            append_result({
                "message_id": message_id,
                "subject": subject,
                "is_selection_email": result.is_selection_email,
                "records": valid_records,
                "prefilter": True,
            })

            all_records.extend(valid_records)
            checkpoint.add(message_id)

            print(
                f"Processed {index}/{len(emails)} -> "
                f"{len(valid_records)} selection records"
            )

        except Exception as exc:
            print(f"ERROR processing {index}/{len(emails)}: {exc}")
            print("This email was NOT checkpointed; rerun will retry it.")

    final_records = deduplicate(all_records)
    count = write_csv(final_records, SELECTION_CSV)
    print(f"Historical selection CSV: {count} rows -> {SELECTION_CSV}")


def test_config():
    print("Email configured:", bool(settings.email_address and settings.email_app_password))
    print("Placement sender configured:", bool(settings.placement_office_email))
    print("Gemini configured:", bool(settings.gemini_api_key))
    print("Gemini model:", settings.gemini_model)


def main():
    parser = argparse.ArgumentParser(
        description="Extract historical college placement selections from TPO emails."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("test-config")

    run_parser = sub.add_parser("run")
    run_parser.add_argument(
        "--limit",
        type=int,
        default=settings.test_email_limit,
        help="Number of latest TPO emails to fetch.",
    )

    args = parser.parse_args()

    if args.command == "test-config":
        test_config()
    elif args.command == "run":
        run(args.limit)


if __name__ == "__main__":
    main()
