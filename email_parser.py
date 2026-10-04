from __future__ import annotations

import email
from email.message import Message
from typing import Any

from email_cleaner import clean_text, html_to_text
from attachment_parser import extract_attachment_text


def _decode_part(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        value = part.get_payload()
        return value if isinstance(value, str) else ""
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def parse_message(msg: Message) -> dict[str, Any]:
    plain_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[dict[str, Any]] = []

    if msg.is_multipart():
        parts = msg.walk()
    else:
        parts = [msg]

    for part in parts:
        disposition = (part.get_content_disposition() or "").lower()
        filename = part.get_filename()

        if disposition == "attachment" or filename:
            data = part.get_payload(decode=True) or b""
            if filename and data:
                extracted = extract_attachment_text(
                    filename, part.get_content_type(), data
                )
                if extracted:
                    attachments.append({
                        "filename": filename,
                        "content_type": part.get_content_type(),
                        "text": clean_text(extracted),
                    })
            continue

        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain_parts.append(_decode_part(part))
        elif content_type == "text/html":
            html_parts.append(_decode_part(part))

    body = "\n\n".join(x for x in plain_parts if x.strip())
    if not body:
        body = "\n\n".join(html_to_text(x) for x in html_parts if x.strip())

    attachment_text = "\n\n".join(
        f"ATTACHMENT: {a['filename']}\n{a['text']}" for a in attachments
    )

    full_text = clean_text(f"{body}\n\n{attachment_text}".strip())

    message_id = msg.get("Message-ID", "").strip()

    return {
        "message_id": message_id,
        "date": msg.get("Date", ""),
        "from": msg.get("From", ""),
        "to": msg.get("To", ""),
        "subject": msg.get("Subject", ""),
        "body": full_text,
        "attachments": attachments,
    }
