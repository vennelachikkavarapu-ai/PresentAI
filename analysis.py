
from faster_whisper import WhisperModel


def transcribe_audio(audio_file):
    """
    Convert an audio recording into text.
    """
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    segments, info = model.transcribe(
        audio_file,
        beam_size=5
    )

    transcript = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return transcript, info.language
