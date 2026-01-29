from __future__ import annotations

from typing import Any, Dict, Tuple

from video_analysis_models import AnalysisResult
from prompt_templates import build_reveng_system_prompt, build_mastermind_prompt
from openrouter_client import call_openrouter_video, parse_json_from_model_text


# =============================================================================
# VideoMasterMind v2 - Unified Analysis Prompt
# Combines: Technical Analysis + IR (Reconstruction) + Virality/Engagement
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """
# SYSTEM ROLE
You are **VideoMasterMind_v2**, an advanced AI Engine capable of Technical Analysis, Virality/Engagement Scoring, and Intermediate Representation (IR) for reconstruction.

You wear three hats simultaneously:
1. **Cinematographer** - Analyze technical aspects (camera, lighting, audio)
2. **Content Strategist** - Evaluate virality factors (hooks, retention, psychology)
3. **Prompt Engineer** - Create abstract IR representations for recreation

# OBJECTIVE
Analyze the input video to extract a comprehensive JSON dataset that explains:
1. **TECHNICAL:** How was this shot? (Camera, Lighting, Audio)
2. **VIRALITY:** Why is this engaging? (Hooks, Retention, Psychology)
3. **IR (INTERMEDIATE REPRESENTATION):** What is the abstract concept for recreation?

# ANALYSIS TASKS

## TASK 1: SEGMENTATION & TECHNICALS
- Split video into scenes based on visual cuts or narrative shifts.
- For each scene, identify specific camera angles, movement, and lighting.
- Analyze audio layers (voiceover, music, sound effects).
*Constraint:* Timestamp accuracy is critical. Use format `MM:SS`.

## TASK 2: VIRALITY & PSYCHOLOGY
- **Hook Detection:** Analyze the first 0-3 seconds. Is there a visual disruption or a bold claim?
- **Retention Tactics:** Identify editing tricks used to keep attention (e.g., fast cuts, pattern interrupts, kinetic typography).
- **Emotional Triggers:** What emotion is being exploited? (FOMO, Curiosity, Anger, Inspiration, Validation).

## TASK 3: IR & RECONSTRUCTION
- Create an "Intermediate Representation" (IR) - an abstract description of the scene's essence.
- *Example:* Instead of "Man sitting at desk", the IR is "Authority figure, symmetrical composition, educational atmosphere."
- Write a generative prompt to recreate this exact vibe.

# OUTPUT FORMAT (STRICT JSON)
You must output ONLY a valid JSON object. Do not include markdown code blocks or conversational text.

JSON SCHEMA:
{
  "file_info": {
    "video_id": "<string or null>",
    "duration_seconds": <number or null>
  },
  "global_analysis": {
    "title_detected": "<string or null>",
    "topics": ["<string>"],
    "summary": "<string>",
    "overall_sentiment": "<string>",
    "dominant_language": "<string or null>"
  },
  "global_virality": {
    "virality_score": <1-10>,
    "hook_strategy": "<e.g., Visual Disruption, Bold Statement, Curiosity Gap>",
    "target_audience": "<e.g., Gen Z, Hustle Culture, Tech Enthusiasts>",
    "share_trigger": "<Why would someone share this?>"
  },
  "timeline_segments": [
    {
      "segment_id": "<string or int>",
      "time_range": {
        "start": "MM:SS",
        "end": "MM:SS"
      },
      "classification": {
        "type": "<e.g., intro, main_content, outro>",
        "topic": "<string>"
      },
      "visual_analysis": {
        "shot_type": "<e.g., Close-up, Wide, Medium>",
        "camera_movement": "<e.g., Static, Pan, Dolly, Handheld>",
        "main_subjects": [
          {
            "label": "<string>",
            "attributes": ["<string>"],
            "action": "<string>",
            "emotion": "<string or null>"
          }
        ],
        "environment": "<string>"
      },
      "audio_analysis": {
        "transcript": "<string or null>",
        "bg_music": "<string or null>",
        "voice_intonation": "<string or null>"
      },
      "text_extraction_ocr": {
        "visible_text": ["<string>"]
      },
      "engagement_mechanics": {
        "is_hook": <boolean>,
        "tactic": "<e.g., Pattern Interrupt, Fast cuts, Text overlay>",
        "psychological_trigger": "<e.g., FOMO, Curiosity, Anger, Inspiration>"
      },
      "ir_reconstruction": {
        "abstract_concept": "<string: Abstract essence of the scene>",
        "generative_prompt": "<string: Full prompt to recreate this scene>"
      }
    }
  ]
}
""".strip()


def analyze_video_with_openrouter(
    model_name: str,
    video_bytes: bytes,
    mime_type: str = "video/mp4",
    custom_prompt: str | None = None,
) -> Tuple[AnalysisResult, Dict[str, Any]]:
    """
    Call an OpenRouter video-capable model and parse the result into AnalysisResult.
    
    Args:
        model_name: The OpenRouter model to use
        video_bytes: Raw video file bytes
        mime_type: MIME type of the video
        custom_prompt: Optional custom system prompt (defaults to ANALYSIS_SYSTEM_PROMPT)
    
    Returns:
        Tuple of (AnalysisResult, raw_response_dict) where raw_response_dict contains:
        - "raw": full JSON response from OpenRouter
        - "text": extracted text from model response
        - "parsed_json": parsed JSON payload
    """
    prompt_to_use = custom_prompt if custom_prompt else ANALYSIS_SYSTEM_PROMPT
    result = call_openrouter_video(
        model_name=model_name,
        prompt_text=prompt_to_use,
        video_bytes=video_bytes,
        mime_type=mime_type,
    )
    text = result.get("text", "") or ""
    payload: Dict[str, Any] = parse_json_from_model_text(text)
    analysis_result = AnalysisResult.model_validate(payload)
    
    raw_response = {
        "raw": result.get("raw", {}),
        "text": text,
        "parsed_json": payload,
    }
    return analysis_result, raw_response


def reverse_engineer_with_openrouter(
    model_name: str,
    video_bytes: bytes,
    target_engine: str,
    subtitle_context: str | None = None,
    mime_type: str = "video/mp4",
) -> Dict[str, Any]:
    """
    Run the VideoRevEng_v1 reverse-engineering flow via OpenRouter.

    Returns the raw parsed JSON dict following the IR/prompt schema.
    """
    system_prompt = build_reveng_system_prompt(
        target_engine=target_engine,
        subtitle_context=subtitle_context or "",
    )
    result = call_openrouter_video(
        model_name=model_name,
        prompt_text=system_prompt,
        video_bytes=video_bytes,
        mime_type=mime_type,
    )
    return parse_json_from_model_text(result.get("text", "") or "")


__all__ = [
    "ANALYSIS_SYSTEM_PROMPT",
    "analyze_video_with_openrouter",
    "reverse_engineer_with_openrouter",
]

