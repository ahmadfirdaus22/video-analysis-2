from __future__ import annotations

import argparse
import os
from typing import Optional

from video_analysis_models import AnalysisResult
from video_llm_integration import (
    analyze_video_structure,
    configure_genai,
    reverse_engineer_scenes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Video analysis & reverse-engineering demo CLI."
    )
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        help="Gemini API key (or set GEMINI_API_KEY / GOOGLE_API_KEY env var).",
    )
    parser.add_argument(
        "--target-engine",
        dest="target_engine",
        default="Kling AI v1.5",
        help="Target video generation engine name used inside prompts.",
    )
    parser.add_argument(
        "--subtitle-file",
        dest="subtitle_file",
        default=None,
        help="Optional subtitle/transcript text file to provide as context.",
    )
    return parser.parse_args()


def load_subtitle_text(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Subtitle file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def demo(video_path: str, api_key: str, target_engine: str, subtitle_file: Optional[str]) -> None:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Configure using explicit key if provided, otherwise fall back to env vars.
    configure_genai(api_key=api_key)

    subtitle_text = load_subtitle_text(subtitle_file)

    # The system prompt for AnalysisResult is assumed to be constructed
    # by the caller; for now we give a minimal, focused instruction that
    # mirrors the AnalysisResult schema.
    analysis_system_prompt = """
You are a Video Content Analysis Engine. Analyze the provided video and output a detailed JSON report.

REQUIREMENTS:
1. Segmentation: Break down the video into logical segments based on visual content changes (e.g., Camera -> Screen Share).
2. Visual Extraction: For each segment, identify the shot type, main subjects, their actions, and emotions.
3. OCR: Extract any legible text on screen, especially code or presentation slides.
4. Audio: Summarize the spoken content and identify the speaker context for each segment.

OUTPUT FORMAT:
Only output a valid JSON object matching this shape:
{
  "file_info": {...},
  "global_analysis": {...},
  "timeline_segments": [...],
  "extracted_entities": {...}
}
""".strip()

    print("Running video analysis (AnalysisResult JSON)...")
    analysis: AnalysisResult = analyze_video_structure(
        video_file_path=video_path,
        system_prompt=analysis_system_prompt,
    )

    print(f"- Segments detected: {len(analysis.timeline_segments)}")
    if analysis.timeline_segments:
        first_segment = analysis.timeline_segments[0]
        ocr_text = (
            first_segment.text_extraction_ocr.visible_text
            if first_segment.text_extraction_ocr
            else []
        )
        print(f"- First segment ID: {first_segment.segment_id}")
        print(f"- First segment OCR text (if any): {ocr_text}")

    print("\nRunning reverse-engineering (IR + prompts)...")
    reveng_result = reverse_engineer_scenes(
        video_file_path=video_path,
        target_engine=target_engine,
        subtitle_data=subtitle_text,
    )

    try:
        timeline = reveng_result.get("timeline", [])
    except AttributeError:
        timeline = []

    if timeline:
        first_scene = timeline[0]
        reconstruction = first_scene.get("reconstruction", {})
        positive_prompt = reconstruction.get("positive_prompt")
        print(f"- First scene positive prompt:\n{positive_prompt}")
    else:
        print("- No scenes returned from reverse-engineering call.")


if __name__ == "__main__":
    args = parse_args()
    demo(
        video_path=args.video_path,
        api_key=args.api_key,
        target_engine=args.target_engine,
        subtitle_file=args.subtitle_file,
    )

