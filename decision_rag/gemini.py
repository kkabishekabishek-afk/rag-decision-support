from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gemini-2.5-flash"
MODEL_FALLBACKS = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-1.5-flash",
)


class GeminiConfigError(RuntimeError):
    pass


class GeminiRequestError(RuntimeError):
    pass


def _setting(name: str, default: str = "" ) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        import streamlit as st
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default

@lru_cache(maxsize=1)
def _client():
    key = _setting("GEMINI_API_KEY")
    if not key or key.lower().startswith("put_your_"):
        raise GeminiConfigError(
            "Gemini is not configured. Add GEMINI_API_KEY to Streamlit Secrets or your local .env file, then restart the app."
        )
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception as exc:
        raise GeminiConfigError(f"Gemini SDK could not be initialized: {exc}") from exc


def model_name() -> str:
    return _setting("GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL


def configured() -> bool:
    key = _setting("GEMINI_API_KEY")
    return bool(key and not key.lower().startswith("put_your_"))


def _is_text_model(name: str, item: Any = None) -> bool:
    """Reject specialized models that cannot return text."""
    low = name.lower()
    blocked = ("tts", "text-to-speech", "embedding", "image", "audio", "robotics", "aqa")
    if any(token in low for token in blocked):
        return False
    methods = getattr(item, "supported_actions", None) or getattr(item, "supported_generation_methods", None) or []
    if methods and not any(str(method).lower() in {"generatecontent", "generate_content"} for method in methods):
        return False
    return True


def _available_models() -> list[str]:
    """Ask Gemini for text-generation models available to this key."""
    try:
        models = _client().models.list()
        available: list[str] = []
        for item in models:
            name = str(getattr(item, "name", "") or "")
            if name.startswith("models/"):
                name = name.removeprefix("models/")
            if name and _is_text_model(name, item):
                available.append(name)
        return available
    except Exception:
        return []


def _model_candidates() -> list[str]:
    requested = model_name()
    # Avoid an extra models-list network request on every fresh process.
    # Known text fallbacks are enough for normal recovery and are faster.
    candidates = [requested, *MODEL_FALLBACKS]
    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen and _is_text_model(candidate):
            seen.add(candidate)
            result.append(candidate)
    return result


@lru_cache(maxsize=1)
def _cached_model_candidates() -> tuple[str, ...]:
    return tuple(_model_candidates())


def _should_retry_model(exc: Exception) -> bool:
    low = str(exc).lower()
    return any(token in low for token in ("404", "not found", "not_found", "invalid_argument", "response modalities", "not supported by the model"))


def _is_temporary_server_error(exc: Exception) -> bool:
    low = str(exc).lower()
    return any(token in low for token in ("500", "internal server", "service unavailable", "503", "deadline exceeded", "timed out"))


def _response_text(response: Any) -> str:
    text = (getattr(response, "text", None) or "").strip()
    if text:
        return text
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            part_text = (getattr(part, "text", None) or "").strip()
            if part_text:
                return part_text
    return ""


def generate(prompt: str, *, temperature: float = 0.1, max_output_tokens: int = 1800) -> str:
    if not prompt.strip():
        raise ValueError("Gemini prompt cannot be empty.")
    try:
        from google.genai import types
        last_error: Exception | None = None
        candidates = list(_cached_model_candidates())[:2]
        for candidate_index, candidate in enumerate(candidates):
            try:
                response = _client().models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                text = _response_text(response)
                if not text:
                    retry_response = _client().models.generate_content(
                        model=candidate,
                        contents=("Return a concise text answer only. Do not return audio, JSON, or an empty response.\n\n" + prompt),
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=min(max_output_tokens, 800),
                        ),
                    )
                    text = _response_text(retry_response)
                if not text:
                    raise GeminiRequestError("Gemini returned an empty text response after retrying.")
                return text
            except Exception as exc:
                last_error = exc
                if _is_temporary_server_error(exc):
                    # A single compact retry handles transient overload without
                    # sending the same large prompt to every fallback model.
                    time.sleep(0.35)
                    try:
                        compact_prompt = "Answer concisely using the supplied context. Return text only.\n\n" + prompt[:12000]
                        retry_response = _client().models.generate_content(
                            model=candidate,
                            contents=compact_prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.0,
                                max_output_tokens=min(max_output_tokens, 900),
                            ),
                        )
                        retry_text = _response_text(retry_response)
                        if retry_text:
                            return retry_text
                    except Exception as retry_exc:
                        last_error = retry_exc
                    if candidate_index == 0:
                        continue
                    break
                if not _should_retry_model(exc):
                    raise
        raise last_error or GeminiRequestError("No text-capable Gemini model is available for this API key.")
    except GeminiConfigError:
        raise
    except Exception as exc:
        msg = str(exc)
        low = msg.lower()
        if any(token in low for token in ("404", "not found", "not_found", "response modalities", "not supported by the model")):
            tried = ", ".join(_cached_model_candidates()[:4]) or "no text-capable models returned"
            raise GeminiRequestError(
                f"No text-capable Gemini model was available for this API key. Tried: {tried}. Choose a text model in Google AI Studio."
            ) from exc
        if _is_temporary_server_error(exc):
            tried = ", ".join(_cached_model_candidates()[:2]) or model_name()
            raise GeminiRequestError(
                f"Gemini is temporarily overloaded. Retried {tried} with a compact request. Please wait a few seconds and try again."
            ) from exc
        if any(token in low for token in ("401", "403", "api key", "unauthenticated", "permission")):
            raise GeminiRequestError("Gemini authentication failed. Check GEMINI_API_KEY.") from exc
        if any(token in low for token in ("429", "quota", "resource_exhausted")):
            raise GeminiRequestError("Gemini quota or rate limit reached. Try again later.") from exc
        raise GeminiRequestError(f"Gemini request failed: {msg}") from exc


def generate_json(prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 1800) -> dict[str, Any]:
    raw = generate(prompt, temperature=temperature, max_output_tokens=max_output_tokens)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise GeminiRequestError("Gemini returned malformed JSON. Please retry the question.") from exc
    if not isinstance(value, dict):
        raise GeminiRequestError("Gemini returned JSON in an unexpected shape.")
    return value
