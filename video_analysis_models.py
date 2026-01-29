from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class TechnicalSpecs(BaseModel):
    resolution: Optional[str] = None
    fps: Optional[float] = None
    audio_channels: Optional[int] = None


class FileInfo(BaseModel):
    video_id: Optional[str] = None
    filename: Optional[str] = None
    processed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    technical_specs: Optional[TechnicalSpecs] = None


class GlobalAnalysis(BaseModel):
    title_detected: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    overall_sentiment: Optional[str] = None
    dominant_language: Optional[str] = None


class TimeRange(BaseModel):
    start: str  # format "HH:MM:SS" or "MM:SS"
    end: str


class Classification(BaseModel):
    type: Optional[str] = None
    topic: Optional[str] = None


class VisualSubject(BaseModel):
    label: Optional[str] = None
    attributes: List[str] = Field(default_factory=list)
    action: Optional[str] = None
    emotion: Optional[str] = None


class VisualAnalysis(BaseModel):
    shot_type: Optional[str] = None
    camera_movement: Optional[str] = None
    main_subjects: List[VisualSubject] = Field(default_factory=list)
    environment: Optional[str] = None


class AudioAnalysis(BaseModel):
    transcript: Optional[str] = None
    speaker_id: Optional[str] = None
    bg_music: Optional[str] = None
    voice_intonation: Optional[str] = None


class TextExtractionOCR(BaseModel):
    visible_text: List[str] = Field(default_factory=list)
    programming_language: Optional[str] = None


# --- VIRALITY & ENGAGEMENT (The "Why") ---
class EngagementMechanics(BaseModel):
    is_hook: bool = False
    tactic: Optional[str] = None  # e.g., "Pattern Interrupt", "Fast cuts", "Text overlay pop-up"
    psychological_trigger: Optional[str] = None  # e.g., "FOMO", "Anger", "Inspiration", "Curiosity"


# --- INTERMEDIATE REPRESENTATION (IR - Abstract) ---
class IRReconstruction(BaseModel):
    abstract_concept: Optional[str] = None  # e.g., "Authority figure, symmetrical composition"
    generative_prompt: Optional[str] = None  # Prompt to recreate the scene with target engine


# --- GLOBAL VIRALITY ANALYSIS ---
class GlobalViralityAnalysis(BaseModel):
    virality_score: Optional[int] = None  # Scale 1-10
    hook_strategy: Optional[str] = None  # e.g., "Visual Disruption", "Bold Statement", "Curiosity Gap"
    target_audience: Optional[str] = None  # e.g., "Gen Z, Hustle Culture"
    share_trigger: Optional[str] = None  # Why would someone share this?

    @field_validator("target_audience", mode="before")
    @classmethod
    def _coerce_target_audience(cls, v):
        """Convert list to comma-separated string if needed."""
        if isinstance(v, list):
            return ", ".join(str(item) for item in v)
        return v


def _seconds_to_time_string(seconds: Union[str, float, int]) -> str:
    """
    Convert seconds (float/int) to time string format "MM:SS" or "HH:MM:SS".
    If input is already a string, return it as-is.
    """
    if isinstance(seconds, str):
        return seconds
    
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


class TimelineSegment(BaseModel):
    segment_id: Optional[str] = None
    time_range: Optional[TimeRange] = None
    # Some models may emit `start` / `end` at the top level instead of nested `time_range`.
    # Accept both string and float/int (seconds) - validator converts to string format
    start: Optional[str] = None
    end: Optional[str] = None
    classification: Optional[Classification] = None
    visual_analysis: Optional[VisualAnalysis] = None
    audio_analysis: Optional[AudioAnalysis] = None
    text_extraction_ocr: Optional[TextExtractionOCR] = None
    # New VideoMasterMind_v2 fields
    engagement_mechanics: Optional[EngagementMechanics] = None
    ir_reconstruction: Optional[IRReconstruction] = None

    @field_validator("segment_id", mode="before")
    @classmethod
    def _coerce_segment_id(cls, v):
        """Convert int to string if needed."""
        if v is None:
            return None
        return str(v)

    @field_validator("text_extraction_ocr", mode="before")
    @classmethod
    def _coerce_text_extraction_ocr(cls, v):
        """Handle empty list or other invalid inputs."""
        if v is None:
            return None
        if isinstance(v, list):
            # Empty list or list of strings -> convert to proper structure
            if len(v) == 0:
                return None
            # If it's a list of strings, wrap it in the expected structure
            return {"visible_text": v}
        return v

    @field_validator("start", "end", mode="before")
    @classmethod
    def _convert_time_to_string(cls, v: Union[str, float, int, None]) -> Optional[str]:
        """
        Convert float/int seconds to time string format before validation.
        """
        if v is None:
            return None
        if isinstance(v, str):
            return v
        return _seconds_to_time_string(v)

    @model_validator(mode="after")
    def _coerce_time_range(self) -> "TimelineSegment":
        """
        Allow either nested `time_range` or top-level `start`/`end` fields.
        """
        if self.time_range is None and self.start is not None and self.end is not None:
            self.time_range = TimeRange(start=self.start, end=self.end)
        return self


class ExtractedAction(BaseModel):
    action: str
    timestamp: str  # format "HH:MM:SS" or "MM:SS"


class ExtractedEntities(BaseModel):
    actions_detected: List[ExtractedAction] = Field(default_factory=list)
    key_objects: List[str] = Field(default_factory=list)
    referenced_links: List[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    file_info: Optional[FileInfo] = None
    global_analysis: Optional[GlobalAnalysis] = None
    global_virality: Optional[GlobalViralityAnalysis] = None  # VideoMasterMind_v2 virality analysis
    timeline_segments: List[TimelineSegment] = Field(default_factory=list)
    extracted_entities: Optional[ExtractedEntities] = None


__all__ = [
    "TechnicalSpecs",
    "FileInfo",
    "GlobalAnalysis",
    "TimeRange",
    "Classification",
    "VisualSubject",
    "VisualAnalysis",
    "AudioAnalysis",
    "TextExtractionOCR",
    "EngagementMechanics",
    "IRReconstruction",
    "GlobalViralityAnalysis",
    "TimelineSegment",
    "ExtractedAction",
    "ExtractedEntities",
    "AnalysisResult",
]

