"""Regex PII redaction to bracketed placeholders."""

from __future__ import annotations

import re

# More-specific patterns first.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "[API_KEY]",
    ),
    (
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?key|bearer)"
            r"\s*[:=]\s*\S+"
        ),
        "[API_KEY]",
    ),
    (
        re.compile(
            r"(?i)\b(?:root\s+)?pass(?:word|wd)?(?:\s+is|\s*[:=])\s*\S+"
        ),
        "[PASSWORD]",
    ),
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL]",
    ),
    (
        re.compile(
            r"\b(?:AD|AE|AL|AT|AZ|BA|BE|BG|BH|BR|BY|CH|CR|CY|CZ|DE|DK|DO|"
            r"EE|EG|ES|FI|FO|FR|GB|GE|GI|GL|GR|GT|HR|HU|IE|IL|IQ|IS|IT|"
            r"JO|KW|KZ|LB|LC|LI|LT|LU|LV|LY|MC|MD|ME|MK|MR|MT|MU|NL|NO|"
            r"PK|PL|PS|PT|QA|RO|RS|SA|SE|SI|SK|SM|SV|TN|TR|UA|VA|VG|XK)"
            r"\d{2}[A-Z0-9]{11,30}\b"
        ),
        "[BANKING_DATA]",
    ),
    (
        re.compile(
            r"(?i)\b(?:routing(?:\s+number)?|iban|card(?:\s+ending\s+in)?|"
            r"cvv|account(?:\s+number)?)\s+[*\dA-Z-]{4,}\b"
        ),
        "[BANKING_DATA]",
    ),
    (
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        "[BANKING_DATA]",
    ),
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[GOV_ID]",
    ),
    (
        re.compile(
            r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)[\s.-]?|\d{3}[\s.-])"
            r"\d{3}[\s.-]?\d{4}\b"
        ),
        "[PHONE]",
    ),
    (
        re.compile(r"(?<!\d)\d{3}[\s.-]\d{4}\b"),
        "[PHONE]",
    ),
    (
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "[CONFIDENTIAL]",
    ),
    (
        re.compile(
            r"\b\d{1,5}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*"
            r"\s+(?:Street|St\.?|Lane|Ln\.?|Avenue|Ave\.?|Road|Rd\.?|"
            r"Boulevard|Blvd\.?)\b"
        ),
        "[ADDRESS]",
    ),
)


def anonymize(text: str) -> str:
    """Replace emails, phones, credentials, and similar tokens with placeholders."""
    redacted = text
    for pattern, placeholder in _RULES:
        redacted = pattern.sub(placeholder, redacted)
    return redacted
