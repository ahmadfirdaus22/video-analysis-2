from __future__ import annotations

import pandas as pd
import streamlit as st

from openrouter_analysis import analyze_video_with_openrouter, ANALYSIS_SYSTEM_PROMPT


# NOTE:
# - Set OPENROUTER_API_KEY in your environment or .env file.
# - Run the app with: `streamlit run streamlit_app.py`


VIDEO_MODELS = [
    # All video-capable models on OpenRouter (as of 2026-01-29)
    # Google Gemini models
    "google/gemini-3-flash-preview",
    "google/gemini-3-pro-preview",
    "google/gemini-2.5-flash-preview-09-2025",
    "google/gemini-2.5-flash-lite-preview-09-2025",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-2.5-pro-preview-05-06",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.0-flash-001",
    # ByteDance Seed models
    "bytedance-seed/seed-1.6-flash",
    "bytedance-seed/seed-1.6",
    # Other providers
    "z-ai/glm-4.6v",
    "amazon/nova-2-lite-v1",
    "nvidia/nemotron-nano-12b-v2-vl",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "allenai/molmo-2-8b:free",
]


def build_segments_dataframe(analysis) -> pd.DataFrame:
    """Build a DataFrame with timeline segments including engagement and IR data."""
    rows = []
    for seg in analysis.timeline_segments:
        classification = seg.classification or {}
        visual = seg.visual_analysis or {}
        ocr = seg.text_extraction_ocr
        engagement = seg.engagement_mechanics or {}
        ir = seg.ir_reconstruction or {}
        
        rows.append(
            {
                "segment_id": seg.segment_id,
                "start": (seg.time_range.start if seg.time_range else getattr(seg, "start", None)),
                "end": (seg.time_range.end if seg.time_range else getattr(seg, "end", None)),
                "type": getattr(classification, "type", None),
                "shot_type": getattr(visual, "shot_type", None),
                "camera": getattr(visual, "camera_movement", None),
                # Engagement mechanics
                "is_hook": getattr(engagement, "is_hook", False),
                "tactic": getattr(engagement, "tactic", None),
                "trigger": getattr(engagement, "psychological_trigger", None),
                # IR
                "ir_concept": getattr(ir, "abstract_concept", None),
            }
        )
    return pd.DataFrame(rows)


def display_virality_metrics(analysis) -> None:
    """Display global virality analysis metrics."""
    virality = analysis.global_virality
    if not virality:
        st.info("No virality analysis available.")
        return
    
    # Virality score as a prominent metric
    score = virality.virality_score or 0
    st.metric("Virality Score", f"{score}/10")
    
    st.markdown("---")
    
    # Detailed info in a clean layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Hook Strategy**")
        st.info(virality.hook_strategy or "N/A")
        
        st.markdown("**Target Audience**")
        st.info(virality.target_audience or "N/A")
    
    with col2:
        st.markdown("**Share Trigger**")
        st.info(virality.share_trigger or "N/A")


def display_ir_prompts(analysis) -> None:
    """Display IR reconstruction prompts for each scene."""
    segments_with_ir = [
        seg for seg in analysis.timeline_segments
        if seg.ir_reconstruction and seg.ir_reconstruction.generative_prompt
    ]
    
    if not segments_with_ir:
        st.info("No IR reconstruction prompts available.")
        return
    
    for seg in segments_with_ir:
        ir = seg.ir_reconstruction
        time_range = seg.time_range
        time_str = f"{time_range.start} - {time_range.end}" if time_range else "N/A"
        
        with st.expander(f"Scene {seg.segment_id} ({time_str})"):
            st.markdown("**Abstract Concept:**")
            st.info(ir.abstract_concept or "N/A")
            
            st.markdown("**Generative Prompt:**")
            st.code(ir.generative_prompt or "N/A", language=None)


def display_engagement_details(analysis) -> None:
    """Display detailed engagement mechanics for each scene."""
    segments_with_engagement = [
        seg for seg in analysis.timeline_segments
        if seg.engagement_mechanics
    ]
    
    if not segments_with_engagement:
        st.info("No engagement mechanics data available.")
        return
    
    for seg in segments_with_engagement:
        eng = seg.engagement_mechanics
        time_range = seg.time_range
        time_str = f"{time_range.start} - {time_range.end}" if time_range else "N/A"
        
        hook_badge = " [HOOK]" if eng.is_hook else ""
        
        with st.expander(f"Scene {seg.segment_id} ({time_str}){hook_badge}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Is Hook:**")
                st.write("Yes" if eng.is_hook else "No")
            
            with col2:
                st.markdown("**Retention Tactic:**")
                st.write(eng.tactic or "N/A")
            
            with col3:
                st.markdown("**Psychological Trigger:**")
                st.write(eng.psychological_trigger or "N/A")


def main() -> None:
    st.set_page_config(page_title="LLM Video Analysis", layout="wide")
    st.title("LLM Video Analysis")
    st.write(
        "Upload a video and get comprehensive analysis including **Technical**, "
        "**Virality/Engagement**, and **IR Reconstruction** data using LLM models."
    )

    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox(
            "OpenRouter model",
            options=VIDEO_MODELS,
            index=0,
        )
        st.caption("API key is read from `OPENROUTER_API_KEY`.")

    uploaded = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "webm", "mpeg"],
        accept_multiple_files=False,
    )

    if uploaded is not None:
        st.write(f"Selected file: **{uploaded.name}**")

    # Editable system prompt
    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "Edit the system prompt below:",
        value=ANALYSIS_SYSTEM_PROMPT,
        height=400,
        help="Modify the prompt to customize the analysis. The model will use this prompt to analyze your video.",
    )

    if st.button("Analyze video", type="primary", disabled=uploaded is None):
        if uploaded is None:
            st.warning("Please upload a video file first.")
            return

        video_bytes = uploaded.read()
        mime_type = uploaded.type or "video/mp4"

        with st.spinner("Analyzing video with LLM models..."):
            try:
                analysis, raw_response = analyze_video_with_openrouter(
                    model_name=model_name,
                    video_bytes=video_bytes,
                    mime_type=mime_type,
                    custom_prompt=system_prompt,
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Analysis failed: {e}")
                return

        # Display cost and token usage at the top
        usage = raw_response.get("raw", {}).get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        cost = usage.get("cost", 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Tokens", f"{total_tokens:,}")
        with col2:
            st.metric("Cost", f"${cost:.6f}")
        
        st.divider()

        # =================================================================
        # SECTION 1: Global Virality Analysis
        # =================================================================
        st.subheader("Global Virality Analysis")
        display_virality_metrics(analysis)
        
        st.divider()

        # =================================================================
        # SECTION 2: Timeline Segments Table
        # =================================================================
        st.subheader("Timeline Segments")
        if analysis.timeline_segments:
            df = build_segments_dataframe(analysis)
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "is_hook": st.column_config.CheckboxColumn("Hook?", default=False),
                    "segment_id": st.column_config.TextColumn("ID"),
                    "start": st.column_config.TextColumn("Start"),
                    "end": st.column_config.TextColumn("End"),
                    "type": st.column_config.TextColumn("Type"),
                    "shot_type": st.column_config.TextColumn("Shot"),
                    "camera": st.column_config.TextColumn("Camera"),
                    "tactic": st.column_config.TextColumn("Retention Tactic"),
                    "trigger": st.column_config.TextColumn("Psych Trigger"),
                    "ir_concept": st.column_config.TextColumn("IR Concept", width="large"),
                },
            )
        else:
            st.info("No timeline segments returned.")
        
        st.divider()

        # =================================================================
        # SECTION 3: Detailed Views (Expandable)
        # =================================================================
        tab1, tab2, tab3 = st.tabs([
            "IR Reconstruction Prompts",
            "Engagement Mechanics Details",
            "Raw JSON Data",
        ])
        
        with tab1:
            st.markdown("**Generative prompts to recreate each scene:**")
            display_ir_prompts(analysis)
        
        with tab2:
            st.markdown("**Detailed engagement analysis per scene:**")
            display_engagement_details(analysis)
        
        with tab3:
            st.markdown("**Full Analysis JSON:**")
            st.json(analysis.model_dump(), expanded=False)
            
            st.markdown("**Raw API Response:**")
            st.json(raw_response, expanded=False)


if __name__ == "__main__":
    main()

