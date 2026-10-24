
from openai import OpenAI


def improve_speech(transcript, api_key, style="Clear and natural"):
    """
    Generate a polished practice version of a speech.
    The original transcript is never overwritten.
    """

    if not transcript or not transcript.strip():
        raise ValueError("The transcript is empty.")

    if not api_key or not api_key.strip():
        raise ValueError("Enter your OpenAI API key first.")

    client = OpenAI(api_key=api_key.strip())

    prompt = f"""
You are PresentAI, a supportive public-speaking coach.

Review the speech transcript below and create a useful practice report.

STYLE: {style}

Return these four sections:

1. CLEANED TRANSCRIPT
Correct punctuation, capitalization, obvious grammar errors,
and clear speech-recognition mistakes where the intended words
are reasonably clear. Preserve the speaker's meaning and ideas.
Do not invent facts. If a word is uncertain, mark it as [unclear]
rather than guessing.

2. IMPROVED PRACTICE SPEECH
Rewrite the speech to sound clear, natural, organized, and suitable
for speaking aloud. Keep the original message and important details.
Use simple, speakable sentences. Do not add unsupported claims.

3. WHAT TO IMPROVE
Give up to five specific, constructive suggestions. Consider
clarity, repetition, filler words, transitions, and structure.
Only comment on issues supported by the transcript.

4. PRACTICE PLAN
Give three short exercises based on this speech, including a
suggested way to practise the opening, main points, and conclusion.

Important:
- Do not claim that grammar correction proves the audio was
  transcribed perfectly.
- Do not treat normal conversational language as automatically wrong.
- Be encouraging, practical, and specific.

TRANSCRIPT:
{transcript}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    result = response.output_text.strip()

    if not result:
        raise ValueError("The AI did not return a speech report.")

    return result
