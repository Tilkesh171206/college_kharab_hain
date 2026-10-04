from __future__ import annotations

import email
import imaplib
from email import policy
from typing import Any

from config import settings
from email_parser import parse_message


class IMAPFetcher:
    def __init__(self):
        self.conn: imaplib.IMAP4_SSL | None = None

    def connect(self):
        if not settings.email_address:
            raise RuntimeError("EMAIL_ADDRESS is missing in .env")
        if not settings.email_app_password:
            raise RuntimeError("EMAIL_APP_PASSWORD is missing in .env")

        self.conn = imaplib.IMAP4_SSL(settings.imap_server, settings.imap_port)
        self.conn.login(settings.email_address, settings.email_app_password)

        status, _ = self.conn.select(settings.mailbox, readonly=True)
        if status != "OK":
            raise RuntimeError(f"Could not select mailbox {settings.mailbox}")

    def _search_criteria(self) -> str:
        if settings.placement_office_email:
            return f'(FROM "{settings.placement_office_email}")'
        return "ALL"

    def fetch(self, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.conn:
            self.connect()

        status, data = self.conn.search(None, self._search_criteria())
        if status != "OK":
            raise RuntimeError("IMAP search failed")

        ids = data[0].split()
        if limit and limit > 0:
            ids = ids[-limit:]

        results = []

        for num in ids:
            try:
                status, msg_data = self.conn.fetch(num, "(RFC822)")
                if status != "OK":
                    continue

                raw = next(
                    (part[1] for part in msg_data if isinstance(part, tuple)),
                    None,
                )
                if not raw:
                    continue

                msg = email.message_from_bytes(raw, policy=policy.default)
                parsed = parse_message(msg)
                parsed["imap_uid"] = num.decode(errors="replace")
                results.append(parsed)

            except Exception as exc:
                print(f"Warning: could not process {num!r}: {exc}")

        return results

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            try:
                self.conn.logout()
            except Exception:
                pass
            self.conn = None
