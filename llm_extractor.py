from __future__ import annotations

import json
import time
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from config import settings
from models import ExtractionResponse
from prompt import SELECTION_EXTRACTION_PROMPT


class GeminiExtractor:
    def __init__(self):
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing in .env")

        self.client = genai.Client(api_key=settings.gemini_api_key)

    def extract(self, email_record: dict[str, Any]) -> ExtractionResponse:
        text = email_record.get("body", "")
        subject = email_record.get("subject", "")
        message_id = email_record.get("message_id", "")

        payload = (
            f"{SELECTION_EXTRACTION_PROMPT}\n\n"
            f"SOURCE SUBJECT:\n{subject}\n\n"
            f"SOURCE MESSAGE ID:\n{message_id}\n\n"
            f"EMAIL CONTENT:\n{text}"
        )

        last_error = None

        for attempt in range(settings.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=settings.gemini_model,
                    contents=payload,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=ExtractionResponse.model_json_schema(),
                    ),
                )

                raw = response.text or "{}"
                data = json.loads(raw)
                return ExtractionResponse.model_validate(data)

            except (json.JSONDecodeError, ValidationError, Exception) as exc:
                last_error = exc
                if attempt == settings.max_retries - 1:
                    break
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Gemini extraction failed: {last_error}")
