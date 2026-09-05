"""Ollama pull and chat wrappers."""

from __future__ import annotations

import logging
from typing import Any

import ollama

logger = logging.getLogger("camoufler")


def pull_model(model: str) -> None:
    """Download model via ollama.pull, logging progress when available."""
    logger.info("Pulling model %s ...", model)
    stream = ollama.pull(model, stream=True)
    for chunk in stream:
        status = chunk.get("status", "")
        if status:
            logger.info("%s", status)


def chat_standardize(
    model: str,
    system_prompt: str,
    user_text: str,
    options: dict[str, Any],
) -> str:
    """Run a single chat turn and return assistant text."""
    logger.debug("System prompt: %s", system_prompt)
    logger.debug("Options: %s", options)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        options=options,
    )
    logger.debug("Raw response: %s", response)
    message = response.get("message") or {}
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Model returned empty content.")
    return content.strip()
