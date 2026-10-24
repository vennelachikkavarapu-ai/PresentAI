
import re

FILLER_WORDS = [
    "um", "uh", "erm", "hmm", "basically",
    "actually", "literally", "you know",
    "i mean", "kind of", "sort of"
]


def analyze_speech(transcript, duration_seconds):
    text = transcript.lower()
    words = re.findall(r"\b[\w']+\b", text)
    word_count = len(words)

    filler_counts = {}

    for filler in FILLER_WORDS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        matches = re.findall(pattern, text)

        if matches:
            filler_counts[filler] = len(matches)

    if duration_seconds > 0:
        speaking_speed = round(
            word_count / (duration_seconds / 60), 1
        )
    else:
        speaking_speed = 0

    suggestions = []

    if word_count < 30:
        suggestions.append(
            "Your speech is quite short. Consider adding "
            "examples or supporting details."
        )

    if speaking_speed == 0:
        suggestions.append(
            "Speaking speed could not be calculated."
        )
    elif speaking_speed < 100:
        suggestions.append(
            "Your pace is relatively slow. Try a slightly "
            "more energetic delivery if appropriate."
        )
    elif speaking_speed > 160:
        suggestions.append(
            "Your pace is relatively fast. Try pausing "
            "between important points."
        )
    else:
        suggestions.append(
            "Your speaking pace is within a commonly used "
            "presentation range."
        )

    filler_total = sum(filler_counts.values())

    if filler_total == 0:
        suggestions.append(
            "No common filler words were detected."
        )
    else:
        suggestions.append(
            f"Detected {filler_total} filler word(s). "
            "Try pausing briefly instead of using fillers."
        )

    return {
        "word_count": word_count,
        "filler_counts": filler_counts,
        "speaking_speed_wpm": speaking_speed,
        "suggestions": suggestions
    }
