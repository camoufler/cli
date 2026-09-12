"""Detect prompt type, rewrite, expand framework slots, anonymize."""

from __future__ import annotations

import logging
from typing import Any

from camoufler.anonymize import anonymize
from camoufler.config import AppConfig
from camoufler.frameworks import TYPE_TO_FRAMEWORK, complete_from_prediction, needs_prediction
from camoufler.ollama_api import chat_classify, chat_predict_slots, chat_standardize
from camoufler.prompt_types import PromptType, classify

logger = logging.getLogger("camoufler")


def _rewrite(
    model: str,
    config: AppConfig,
    user_text: str,
    options: dict[str, Any],
    *,
    required: bool,
) -> str:
    """Rewrite into standard English. Prompt path falls back to original on failure."""
    try:
        rewritten = chat_standardize(model, config.system_prompt, user_text, options)
    except Exception:
        if required:
            raise
        logger.info("Rewrite failed; expanding from original text.")
        return user_text.strip()
    cleaned = rewritten.strip()
    return cleaned or user_text.strip()


def run_standardize(
    model: str,
    config: AppConfig,
    user_text: str,
    options: dict[str, Any],
) -> str:
    """Classify, rewrite to standard English, then expand prompts or return the rewrite."""
    prompt_type = classify(
        user_text,
        model=model,
        options=options,
        classify_chat=chat_classify,
    )
    logger.info("Detected prompt type: %s", prompt_type.value)
    cleaned = _rewrite(
        model,
        config,
        user_text,
        options,
        required=prompt_type is PromptType.UTTERANCE,
    )
    if prompt_type is PromptType.UTTERANCE:
        return anonymize(cleaned)

    framework = TYPE_TO_FRAMEWORK[prompt_type.value]
    predicted = ""
    if needs_prediction(framework, cleaned):
        predicted = chat_predict_slots(
            model, framework.system_prompt, cleaned, options
        )
    paragraph = complete_from_prediction(framework, cleaned, predicted or None)
    return anonymize(paragraph)
