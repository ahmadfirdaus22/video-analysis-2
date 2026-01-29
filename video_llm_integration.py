from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import google.generativeai as genai  # type: ignore[import]
from dotenv import load_dotenv  # type: ignore[import]

from video_analysis_models import AnalysisResult
from prompt_templates import build_reveng_system_prompt


def configure_genai(api_key: Optional[str] = None) -> None:
    """
    Configure the google-generativeai client.

    - Loads environment variables from a local `.env` file (if present).
    - If `api_key` is not provided, attempts to read it from `GEMINI_API_KEY`
      or `GOOGLE_API_KEY`.
    """
    # Load from .env so local development can just define keys there.
    load_dotenv()

    effective_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not effective_key:
        raise RuntimeError(
            "No API key provided. Set GEMINI_API_KEY/GOOGLE_API_KEY or "
            "pass api_key explicitly to configure_genai()."
        )
    genai.configure(api_key=effective_key)


def _parse_json_safe(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON parsing helper with simple cleanup for common artifacts.
    """
    text = text.strip()

    # Remove leading markdown fences if the model still produces them.
    if text.startswith("```"):
        # Strip first fence
        first_fence_end = text.find("\n")
        if first_fence_end != -1:
            text = text[first_fence_end + 1 :]
        # Strip possible trailing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON output", "raw": text}


def analyze_video_structure(
    video_file_path: str,
    system_prompt: str,
    model_name: str = "models/gemini-1.5-flash",
) -> AnalysisResult:
    """
    Generic video analysis call that expects the model to return AnalysisResult JSON.

    The `system_prompt` should instruct the model to emit JSON matching `AnalysisResult`.
    """
    model = genai.GenerativeModel(model_name)

    video_file = genai.upload_file(path=video_file_path)

    response = model.generate_content(
        [system_prompt, video_file],
        generation_config={"response_mime_type": "application/json"},
    )

    payload = _parse_json_safe(getattr(response, "text", "") or "")
    # Let Pydantic validate/normalize into our schema.
    return AnalysisResult.model_validate(payload)


def reverse_engineer_scenes(
    video_file_path: str,
    target_engine: str,
    subtitle_data: Optional[str] = None,
    model_name: str = "models/gemini-1.5-flash",
) -> Dict[str, Any]:
    """
    Run the VideoRevEng_v1 reverse-engineering flow.

    Returns the raw parsed JSON dict following the IR/prompt schema
    defined in the system prompt.
    """
    system_prompt = build_reveng_system_prompt(
        target_engine=target_engine,
        subtitle_context=subtitle_data or "",
    )

    model = genai.GenerativeModel(model_name)
    video_file = genai.upload_file(path=video_file_path)

    response = model.generate_content(
        [system_prompt, video_file],
        generation_config={"response_mime_type": "application/json"},
    )

    return _parse_json_safe(getattr(response, "text", "") or "")


__all__ = [
    "configure_genai",
    "analyze_video_structure",
    "reverse_engineer_scenes",
]

