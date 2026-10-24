
import os
import tempfile

import av
import numpy as np
import pandas as pd
import streamlit as st

from analysis import transcribe_audio
from feedback import analyze_speech
from database import create_table, save_session, get_history


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="PresentAI Dashboard",
    page_icon="🎤",
    layout="wide"
)

# =========================================================
# INITIALIZE DATABASE AND SESSION STATE
# =========================================================
create_table()

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "session_saved" not in st.session_state:
    st.session_state.session_saved = False


# =========================================================
# SMART PAUSE DETECTOR
# =========================================================
def detect_speech_pauses(
    audio_path,
    min_pause_seconds=0.4,
    silence_threshold_db=-35
):
    """Detect quiet audio frames and group them into pauses."""

    frame_data = []
    elapsed_fallback = 0.0

    with av.open(audio_path) as container:
        audio_stream = next(
            (
                stream
                for stream in container.streams
                if stream.type == "audio"
            ),
            None
        )

        if audio_stream is None:
            raise ValueError("No audio stream was found.")

        for frame in container.decode(audio_stream):
            sample_rate = frame.sample_rate or audio_stream.rate

            if not sample_rate:
                continue

            original_samples = frame.to_ndarray()

            if original_samples.size == 0:
                continue

            samples = original_samples.astype(np.float64)

            # Normalize integer audio to approximately -1 to 1.
            if np.issubdtype(original_samples.dtype, np.integer):
                max_value = float(
                    max(abs(np.iinfo(original_samples.dtype).min),
                        np.iinfo(original_samples.dtype).max)
                )
                samples = samples / max_value

            rms = float(np.sqrt(np.mean(np.square(samples))))
            dbfs = 20 * np.log10(max(rms, 1e-8))

            frame_duration = (
                frame.samples / sample_rate
                if frame.samples
                else samples.size / sample_rate
            )

            start_time = (
                float(frame.time)
                if frame.time is not None
                else elapsed_fallback
            )

            end_time = start_time + frame_duration
            elapsed_fallback = max(elapsed_fallback, end_time)

            frame_data.append({
                "start": start_time,
                "end": end_time,
                "dbfs": float(dbfs)
            })

    if not frame_data:
        raise ValueError("No readable audio frames were found.")

    timeline_df = pd.DataFrame(frame_data)
    timeline_df["Time (s)"] = timeline_df["start"].round(3)
    timeline_df["Audio Level (dBFS)"] = timeline_df["dbfs"].round(2)

    pauses = []
    pause_start = None
    pause_end = None

    for frame in frame_data:
        is_quiet = frame["dbfs"] < silence_threshold_db

        if is_quiet:
            if pause_start is None:
                pause_start = frame["start"]

            pause_end = frame["end"]

        elif pause_start is not None:
            duration = pause_end - pause_start

            if duration >= min_pause_seconds:
                pauses.append({
                    "Start (s)": round(pause_start, 2),
                    "End (s)": round(pause_end, 2),
                    "Duration (s)": round(duration, 2)
                })

            pause_start = None
            pause_end = None

    # Include a quiet period that continues to the end.
    if pause_start is not None:
        duration = pause_end - pause_start

        if duration >= min_pause_seconds:
            pauses.append({
                "Start (s)": round(pause_start, 2),
                "End (s)": round(pause_end, 2),
                "Duration (s)": round(duration, 2)
            })

    return pauses, timeline_df


def format_time(seconds):
    """Format seconds as minutes:seconds."""
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def build_pause_summary(pauses):
    """Summarize detected quiet periods."""

    if not pauses:
        return {
            "count": 0,
            "total": 0.0,
            "longest": 0.0,
            "average": 0.0
        }

    durations = [p["Duration (s)"] for p in pauses]

    return {
        "count": len(pauses),
        "total": round(sum(durations), 2),
        "longest": round(max(durations), 2),
        "average": round(sum(durations) / len(durations), 2)
    }


# =========================================================
# PRACTICE HISTORY HELPERS
# =========================================================
def load_history_dataframe():
    """Convert database history into a DataFrame safely."""

    history = get_history()

    if history is None:
        return pd.DataFrame()

    if isinstance(history, pd.DataFrame):
        return history.copy()

    if isinstance(history, list):
        if not history:
            return pd.DataFrame()

        if isinstance(history[0], dict):
            return pd.DataFrame(history)

        # Support tuple rows returned from SQLite.
        expected_columns = [
            "id",
            "date",
            "word_count",
            "speaking_speed",
            "filler_count",
            "practice_score",
            "transcript"
        ]

        if isinstance(history[0], (tuple, list)):
            if len(history[0]) == len(expected_columns):
                return pd.DataFrame(
                    history,
                    columns=expected_columns
                )

            without_id = expected_columns[1:]
            if len(history[0]) == len(without_id):
                return pd.DataFrame(
                    history,
                    columns=without_id
                )

        return pd.DataFrame(history)

    return pd.DataFrame(history)


# =========================================================
# HEADER
# =========================================================
st.title("🎤 PresentAI")
st.subheader("Your Intelligent Public Speaking Coach")

st.write(
    "Analyze your speech, understand your speaking habits, "
    "track your progress, and discover ways to improve "
    "your presentations."
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("🎤 PresentAI")
    st.write("Presentation Analyzer")

    st.info(
        "Upload a recording for transcription, speaking metrics, "
        "pause detection, and improvement suggestions."
    )

    st.caption(
        "Practice sessions are saved in a local SQLite database."
    )

    if st.button("🔄 Clear Current Results"):
        st.session_state.analysis_result = None
        st.session_state.session_saved = False
        st.rerun()


# =========================================================
# AUDIO UPLOAD AND SETTINGS
# =========================================================
st.header("📁 Upload Presentation")

audio_file = st.file_uploader(
    "Choose an audio recording",
    type=["wav", "mp3", "m4a", "ogg", "flac"]
)

st.subheader("⚙️ Pause Detection Settings")

setting_col1, setting_col2 = st.columns(2)

with setting_col1:
    min_pause_seconds = st.slider(
        "Minimum pause duration (seconds)",
        min_value=0.2,
        max_value=2.0,
        value=0.4,
        step=0.1
    )

with setting_col2:
    silence_threshold_db = st.slider(
        "Silence threshold (dBFS)",
        min_value=-60,
        max_value=-15,
        value=-35,
        step=1
    )


# =========================================================
# ANALYZE AUDIO
# =========================================================
if audio_file is not None:
    st.audio(audio_file)

    if st.button(
        "Analyze My Presentation",
        type="primary",
        use_container_width=True
    ):
        temp_path = None

        try:
            suffix = os.path.splitext(audio_file.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(audio_file.getvalue())
                temp_path = temp_file.name

            # Audio duration
            with av.open(temp_path) as media:
                if media.duration is None:
                    raise ValueError(
                        "Could not determine audio duration."
                    )

                duration_seconds = media.duration / av.time_base

            if duration_seconds <= 0:
                raise ValueError("The audio duration is invalid.")

            # Transcription
            with st.spinner("Transcribing your presentation..."):
                transcript, language = transcribe_audio(temp_path)

            if not transcript or not transcript.strip():
                st.warning(
                    "No speech was detected. Try another recording."
                )

            else:
                # Speech metrics
                results = analyze_speech(
                    transcript,
                    duration_seconds
                )

                word_count = results["word_count"]
                speed = results["speaking_speed_wpm"]
                filler_counts = results["filler_counts"]
                suggestions = results.get("suggestions", [])

                total_fillers = sum(filler_counts.values())

                # Pause detection
                with st.spinner("Detecting quiet pauses..."):
                    pauses, timeline_df = detect_speech_pauses(
                        temp_path,
                        min_pause_seconds=min_pause_seconds,
                        silence_threshold_db=silence_threshold_db
                    )

                pause_summary = build_pause_summary(pauses)

                # Rule-based practice score
                score = 100

                if speed < 100 or speed > 160:
                    score -= 20

                score -= min(total_fillers * 5, 30)

                if word_count < 30:
                    score -= 10

                score = max(0, min(100, score))

                # Store results so they survive Streamlit reruns.
                st.session_state.analysis_result = {
                    "transcript": transcript,
                    "language": language,
                    "duration_seconds": duration_seconds,
                    "word_count": word_count,
                    "speed": speed,
                    "filler_counts": filler_counts,
                    "total_fillers": total_fillers,
                    "suggestions": suggestions,
                    "pauses": pauses,
                    "timeline_df": timeline_df,
                    "pause_summary": pause_summary,
                    "score": score,
                    "min_pause_seconds": min_pause_seconds,
                    "silence_threshold_db": silence_threshold_db
                }

                st.session_state.session_saved = False

                st.success("Analysis completed successfully!")

        except Exception as error:
            st.error(f"Processing failed: {error}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

else:
    st.info("Upload an audio recording above to get started.")


# =========================================================
# DISPLAY ANALYSIS RESULTS
# This section is outside the Analyze button so results persist.
# =========================================================
data = st.session_state.analysis_result

if data is not None:
    transcript = data["transcript"]
    language = data["language"]
    duration_seconds = data["duration_seconds"]
    word_count = data["word_count"]
    speed = data["speed"]
    filler_counts = data["filler_counts"]
    total_fillers = data["total_fillers"]
    suggestions = data["suggestions"]
    pauses = data["pauses"]
    timeline_df = data["timeline_df"]
    pause_summary = data["pause_summary"]
    score = data["score"]

    st.divider()
    st.header("📊 Your Speaking Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Words Spoken", word_count)
    col2.metric("Speaking Speed", f"{speed} WPM")
    col3.metric("Practice Score", f"{score}/100")

    st.caption(
        "The score is a simple rule-based indicator based on pace, "
        "filler words, and transcript length. It is not a validated "
        "measure of presentation quality."
    )

    st.progress(score / 100)

    # -----------------------------------------------------
    # SMART PAUSE DETECTOR
    # -----------------------------------------------------
    st.divider()
    st.header("⏸️ Smart Pause Detector")

    st.write(
        "Review quiet gaps in your recording. Pauses can help "
        "emphasize important ideas, so the goal is not to eliminate "
        "every pause."
    )

    p1, p2, p3, p4 = st.columns(4)

    p1.metric("Pauses Detected", pause_summary["count"])
    p2.metric("Total Quiet Time", f'{pause_summary["total"]:.2f} s')
    p3.metric("Longest Pause", f'{pause_summary["longest"]:.2f} s')
    p4.metric("Average Pause", f'{pause_summary["average"]:.2f} s')

    st.caption(
        f"Settings: minimum pause {data['min_pause_seconds']:.1f} s; "
        f"silence threshold {data['silence_threshold_db']} dBFS."
    )

    st.subheader("🔊 Audio Level Timeline")

    st.line_chart(
        timeline_df,
        x="Time (s)",
        y="Audio Level (dBFS)"
    )

    st.caption(
        "Lower values indicate quieter audio. Background noise and "
        "recording volume can affect pause detection."
    )

    st.subheader("📍 Detected Pause Timeline")

    if pauses:
        pause_df = pd.DataFrame(pauses)
        pause_df["Start Time"] = pause_df["Start (s)"].apply(format_time)
        pause_df["End Time"] = pause_df["End (s)"].apply(format_time)

        st.dataframe(
            pause_df[
                [
                    "Start Time",
                    "End Time",
                    "Start (s)",
                    "End (s)",
                    "Duration (s)"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "No pauses matched these settings. Try lowering the "
            "minimum duration or adjusting the silence threshold."
        )

    # -----------------------------------------------------
    # SAVE PRACTICE SESSION
    # Outside the Analyze button; this fixes the rerun issue.
    # -----------------------------------------------------
    st.divider()
    st.header("💾 Save Practice Session")

    if st.session_state.session_saved:
        st.success("This practice session has already been saved.")
    else:
        st.write(
            "Save your results to Practice History so you can "
            "compare your future presentations."
        )

        if st.button(
            "Save This Session",
            type="primary",
            key="save_practice_session"
        ):
            try:
                save_session(
                    word_count=word_count,
                    speaking_speed=speed,
                    filler_count=total_fillers,
                    practice_score=score,
                    transcript=transcript
                )

                st.session_state.session_saved = True
                st.success("Practice session saved successfully!")
                st.rerun()

            except Exception as error:
                st.error(f"Could not save session: {error}")

    # -----------------------------------------------------
    # SPEAKING ANALYTICS
    # -----------------------------------------------------
    st.divider()
    st.header("📈 Speaking Analytics")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Speaking Speed")

        speed_data = pd.DataFrame({
            "Metric": ["Your speaking speed"],
            "Words per minute": [speed]
        })

        st.bar_chart(
            speed_data,
            x="Metric",
            y="Words per minute"
        )

        st.caption(
            "Around 100–160 words per minute is a general reference, "
            "not a strict rule."
        )

    with chart_col2:
        st.subheader("Filler Word Frequency")

        if filler_counts:
            filler_data = pd.DataFrame({
                "Filler Word": list(filler_counts.keys()),
                "Count": list(filler_counts.values())
            })

            st.bar_chart(
                filler_data,
                x="Filler Word",
                y="Count"
            )
        else:
            st.success("No tracked filler words detected.")

    # -----------------------------------------------------
    # TRANSCRIPT AND FILLER ANALYSIS
    # -----------------------------------------------------
    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("📝 Speech Transcript")
        st.write(f"**Detected Language:** {language}")
        st.write(transcript)

        st.download_button(
            "📥 Download Transcript",
            data=transcript,
            file_name="presentai_transcript.txt",
            mime="text/plain"
        )

    with right:
        st.subheader("🗣️ Filler Word Analysis")

        if filler_counts:
            for word, count in filler_counts.items():
                st.write(f"- **{word}**: {count}")

            st.write(f"**Total filler words:** {total_fillers}")
        else:
            st.success("No tracked filler words detected.")

    # -----------------------------------------------------
    # IMPROVEMENT SUGGESTIONS
    # -----------------------------------------------------
    st.divider()
    st.header("💡 Improvement Suggestions")

    if suggestions:
        for suggestion in suggestions:
            st.info(suggestion)
    else:
        st.write("Keep practicing and reviewing your recordings.")

    if pause_summary["longest"] >= 2:
        st.info(
            "Pause tip: listen to your longest quiet gap and decide "
            "whether it adds emphasis or interrupts your message."
        )
    elif pause_summary["count"] > 0:
        st.info(
            "Pause tip: listen to the detected gaps and check whether "
            "your pauses feel intentional."
        )
    else:
        st.info(
            "Pause tip: natural pauses can help listeners follow your "
            "ideas. You do not need to eliminate every quiet moment."
        )

    # -----------------------------------------------------
    # DOWNLOAD FULL REPORT
    # -----------------------------------------------------
    filler_report = (
        "\n".join(
            f"{word}: {count}"
            for word, count in filler_counts.items()
        )
        if filler_counts
        else "No tracked filler words detected."
    )

    pause_report = (
        "\n".join(
            f"Start: {p['Start (s)']} s | "
            f"End: {p['End (s)']} s | "
            f"Duration: {p['Duration (s)']} s"
            for p in pauses
        )
        if pauses
        else "No pauses matched the current settings."
    )

    suggestion_report = (
        "\n".join(f"- {item}" for item in suggestions)
        if suggestions
        else "Continue practicing."
    )

    report = f"""PresentAI - Presentation Analysis Report

Detected Language: {language}
Word Count: {word_count}
Speaking Speed: {speed} words per minute
Practice Score: {score}/100
Audio Duration: {duration_seconds:.1f} seconds

Smart Pause Detector:
Pauses Detected: {pause_summary['count']}
Total Quiet Time: {pause_summary['total']} seconds
Longest Pause: {pause_summary['longest']} seconds
Average Pause: {pause_summary['average']} seconds

Pause Timeline:
{pause_report}

Filler Word Counts:
{filler_report}

Transcript:
{transcript}

Improvement Suggestions:
{suggestion_report}
"""

    st.download_button(
        "📄 Download Full Analysis Report",
        data=report,
        file_name="presentai_report.txt",
        mime="text/plain"
    )


# =========================================================
# PRACTICE HISTORY
# Always displayed, even when no audio is uploaded.
# =========================================================
st.divider()
st.header("📚 Practice History")

try:
    history_df = load_history_dataframe()

    if history_df.empty:
        st.info(
            "No saved sessions yet. Upload a recording, analyze it, "
            "and click Save This Session."
        )

    else:
        # Make sure required fields exist before displaying history.
        required_columns = [
            "date",
            "word_count",
            "speaking_speed",
            "filler_count",
            "practice_score"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in history_df.columns
        ]

        if missing_columns:
            st.error(
                "Your database history is missing these columns: "
                + ", ".join(missing_columns)
                + ". Check database.py and its table definition."
            )

        else:
            # Normalize numbers for metrics and charts.
            for column in [
                "word_count",
                "speaking_speed",
                "filler_count",
                "practice_score"
            ]:
                history_df[column] = pd.to_numeric(
                    history_df[column],
                    errors="coerce"
                )

            history_df = history_df.dropna(
                subset=["practice_score"]
            ).reset_index(drop=True)

            if history_df.empty:
                st.info(
                    "No valid practice scores are available in the database."
                )
            else:
                # SQLite history is usually newest first.
                if "date" in history_df.columns:
                    history_df["date"] = history_df["date"].astype(str)

                total_sessions = len(history_df)
                average_score = history_df["practice_score"].mean()
                latest_score = history_df.iloc[0]["practice_score"]

                hist_col1, hist_col2, hist_col3 = st.columns(3)

                hist_col1.metric("Sessions Saved", total_sessions)
                hist_col2.metric(
                    "Average Practice Score",
                    f"{average_score:.1f}/100"
                )
                hist_col3.metric(
                    "Latest Practice Score",
                    f"{latest_score:.0f}/100"
                )

                st.subheader("Previous Sessions")

                display_columns = [
                    "date",
                    "word_count",
                    "speaking_speed",
                    "filler_count",
                    "practice_score"
                ]

                history_display = history_df[
                    display_columns
                ].rename(
                    columns={
                        "date": "Date",
                        "word_count": "Words Spoken",
                        "speaking_speed": "Speaking Speed (WPM)",
                        "filler_count": "Filler Words",
                        "practice_score": "Practice Score"
                    }
                )

                st.dataframe(
                    history_display,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("📈 Score Progress")

                chart_df = history_df.copy()
                chart_df["date_parsed"] = pd.to_datetime(
                    chart_df["date"],
                    errors="coerce"
                )

                chart_df = chart_df.dropna(
                    subset=["date_parsed"]
                ).sort_values("date_parsed")

                if not chart_df.empty:
                    chart_df["Session"] = range(
                        1,
                        len(chart_df) + 1
                    )

                    chart_data = chart_df.set_index("Session")[
                        ["practice_score"]
                    ].rename(
                        columns={"practice_score": "Practice Score"}
                    )

                    st.line_chart(chart_data)

                    st.caption(
                        "Sessions are plotted from oldest to newest."
                    )

                # Show the latest transcript if the database includes it.
                if "transcript" in history_df.columns:
                    st.subheader("🔎 Latest Saved Session")

                    latest = history_df.iloc[0]

                    st.write(f"**Date:** {latest['date']}")
                    st.write(
                        f"**Words spoken:** {latest['word_count']}"
                    )
                    st.write(
                        f"**Speaking speed:** "
                        f"{latest['speaking_speed']:.1f} WPM"
                    )
                    st.write(
                        f"**Filler words:** {latest['filler_count']}"
                    )
                    st.write(
                        f"**Practice score:** "
                        f"{latest['practice_score']:.0f}/100"
                    )

                    with st.expander("View latest saved transcript"):
                        st.write(latest["transcript"])

except Exception as error:
    st.error(f"Could not load practice history: {error}")
    st.caption(
        "If this continues, check the get_history() function and "
        "the table columns in database.py."
    )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "PresentAI | Practice • Improve • Present with Confidence"
)
