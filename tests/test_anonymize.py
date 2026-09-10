"""Tests for regex PII redaction."""

from camoufler.anonymize import anonymize


def test_redacts_email_and_aws_key():
    text = (
        "Ping Alice at asmith@initech.corp. Her AWS access key is "
        "AKIAIOSFODNN7EXAMPLE and root pass is Summer2024!"
    )
    out = anonymize(text)
    assert "asmith@initech.corp" not in out
    assert "[EMAIL]" in out
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "[API_KEY]" in out
    assert "Summer2024!" not in out
    assert "[PASSWORD]" in out


def test_redacts_phone_and_address():
    text = "Call me at 555-0107 or stop by 123 Example Lane."
    out = anonymize(text)
    assert "555-0107" not in out
    assert "[PHONE]" in out
    assert "123 Example Lane" not in out
    assert "[ADDRESS]" in out


def test_redacts_iban_and_routing():
    text = (
        "Refund to IBAN GB29XAPI40151598765432 or routing number 021000021."
    )
    out = anonymize(text)
    assert "GB29XAPI40151598765432" not in out
    assert "021000021" not in out
    assert "[BANKING_DATA]" in out


def test_redacts_ssn_and_ipv4():
    text = "SSN 123-45-6789 on host 10.0.0.8."
    out = anonymize(text)
    assert "123-45-6789" not in out
    assert "[GOV_ID]" in out
    assert "10.0.0.8" not in out
    assert "[CONFIDENTIAL]" in out


def test_leaves_plain_text():
    assert anonymize("I will leave later.") == "I will leave later."
