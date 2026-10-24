
import json
from openai import OpenAI


def analyze_speech_structure(transcript, api_key):
    """Analyze a speech transcript and return structured feedback."""

    if not transcript or not transcript.strip():
        raise ValueError("The transcript is empty. Analyze your audio first.")

    if not api_key or not api_key.strip():
        raise ValueError("Please enter your OpenAI API key.")

    client = OpenAI(api_key=api_key.strip())

    prompt = """
You are PresentAI, a supportive public-speaking coach.

Analyze the supplied presentation transcript using these five categories:
1. Introduction: topic, purpose, opening, audience engagement.
2. Main points: identifiable ideas, organization, logical order.
3. Supporting details: examples, explanations, evidence.
4. Transitions: connections and flow between ideas.
5. Conclusion: summary, final takeaway, closing.

Instructions:
- Base feedback only on the transcript.
- Never invent facts or quotations.
- A short speech should not automatically score poorly.
- If evidence is insufficient, state that clearly.
- Scores are estimates, not objective grades.
- Give specific, constructive suggestions.
- Return only valid JSON, with no Markdown fences.

Use this exact JSON structure:
{
  "overall_score": 0,
  "summary": "Brief assessment",
  "sections": [
    {
      "name": "Introduction",
      "score": 0,
      "evidence": "Evidence from transcript",
      "suggestion": "Practical improvement"
    },
    {
      "name": "Main Points",
      "score": 0,
      "evidence": "Evidence from transcript",
      "suggestion": "Practical improvement"
    },
    {
      "name": "Supporting Details",
      "score": 0,
      "evidence": "Evidence from transcript",
      "suggestion": "Practical improvement"
    },
    {
      "name": "Transitions",
      "score": 0,
      "evidence": "Evidence from transcript",
      "suggestion": "Practical improvement"
    },
    {
      "name": "Conclusion",
      "score": 0,
      "evidence": "Evidence from transcript",
      "suggestion": "Practical improvement"
    }
  ],
  "strengths": ["Strength 1", "Strength 2"],
  "improvements": ["Improvement 1", "Improvement 2"],
  "suggested_outline": [
    "Introduction: ...",
    "Main point 1: ...",
    "Main point 2: ...",
    "Conclusion: ..."
  ]
}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=prompt,
        input="Analyze this transcript:\n\n" + transcript,
    )

    raw_result = response.output_text.strip()

    if not raw_result:
        raise ValueError("The AI returned an empty analysis.")

    try:
        report = json.loads(raw_result)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The AI returned an invalid report. Please try again."
        ) from exc

    # Validate the report before displaying it.
    overall_score = report.get("overall_score")

    if (
        not isinstance(overall_score, int)
        or isinstance(overall_score, bool)
        or not 0 <= overall_score <= 100
    ):
        raise ValueError("The overall score is invalid.")

    sections = report.get("sections")

    if not isinstance(sections, list):
        raise ValueError("The section analysis is invalid.")

    for section in sections:
        if not isinstance(section, dict):
            raise ValueError("A section in the report is invalid.")

        score = section.get("score")

        if (
            not isinstance(score, int)
            or isinstance(score, bool)
            or not 0 <= score <= 100
        ):
            raise ValueError("A section score is invalid.")

        for field in ("name", "evidence", "suggestion"):
            if not isinstance(section.get(field), str):
                raise ValueError(f"Invalid section field: {field}")

    return report
