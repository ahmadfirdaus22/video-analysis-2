from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict

import requests
from dotenv import load_dotenv  # type: ignore[import]


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_openrouter_api_key() -> str:
    """
    Load the OpenRouter API key from environment or .env file.

    Expected variable: OPENROUTER_API_KEY
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Define it in your environment or .env file."
        )
    return api_key


def _encode_video_to_data_url(video_bytes: bytes, mime_type: str = "video/mp4") -> str:
    """
    Encode raw video bytes into a base64 data URL accepted by OpenRouter.
    """
    b64 = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def call_openrouter_video(
    model_name: str,
    prompt_text: str,
    video_bytes: bytes,
    mime_type: str = "video/mp4",
) -> Dict[str, Any]:
    """
    Call an OpenRouter video-capable model with a text + video message.

    Returns a dict with at least:
      - `raw`: the full JSON response from OpenRouter
      - `text`: the first text segment from the model's reply (if any)
    """
    api_key = load_openrouter_api_key()

    data_url = _encode_video_to_data_url(video_bytes, mime_type=mime_type)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "video_url",
                    "video_url": {"url": data_url},
                },
            ],
        }
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": messages,
    }

    resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=300)
    resp.raise_for_status()
    resp_json = resp.json()

    # Extract first text response, following OpenAI-style schema.
    text_out = ""
    try:
        choice = resp_json["choices"][0]
        content = choice["message"]["content"]
        # content is typically a list of segments; grab the first text item
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text_out = part.get("text", "")
                    if text_out:
                        break
        elif isinstance(content, str):
            text_out = content
    except (KeyError, IndexError, TypeError):
        text_out = ""

    return {"raw": resp_json, "text": text_out}


def parse_json_from_model_text(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON parsing with simple cleanup for common artifacts.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_fence_end = cleaned.find("\n")
        if first_fence_end != -1:
            cleaned = cleaned[first_fence_end + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON output", "raw": cleaned}


__all__ = [
    "load_openrouter_api_key",
    "call_openrouter_video",
    "parse_json_from_model_text",
]

