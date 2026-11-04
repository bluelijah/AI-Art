"""Sentence-level rhetoric and emotion tags from an LLM.

The model sees the transcript only (never the audio or the audience), so
`invites_applause` is a genuine prediction we can test against the
detected applause.

Runs through the Claude API with structured outputs: the response is
validated against `SpeechAnnotation`, so every sentence comes back with
the same fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
MODEL = "claude-opus-5-5"

Device = Literal["CONTRAST", "LIST_OF_THREE", "ANAPHORA", "RHETORICAL_QUESTION", "DIRECT_ADDRESS",
                 "STATISTIC", "PERSONAL_TESTIMONY", "ACCUSATION", "PUZZLE_SOLUTION", "WARNING",
                 "PUNCHLINE", "NONE"]
Emotion = Literal["ANGER", "GRIEF", "FEAR", "DEFIANCE", "RESOLVE", "DISBELIEF", "IRONY", "HOPE", "NEUTRAL"]


class SentenceAnnotation(BaseModel):
    seg_id: int
    devices: list[Device]
    emotion: Emotion
    intensity: int = Field(ge=0, le=3)
    addressee: Literal["LEADERS", "YOUTH", "WORLD", "SELF", "NONE"]
    topic: Literal["PERSONAL_STAKES", "SCIENCE_NUMBERS", "INACTION", "MORAL_JUDGEMENT",
                   "FUTURE_WARNING", "GREETING_CLOSING", "OTHER"]
    invites_applause: bool
    note: str


class SpeechAnnotation(BaseModel):
    sentences: list[SentenceAnnotation]


def build_user_message(segments: pd.DataFrame, context: str) -> str:
    lines = [f"[{r.seg_id}] {r.text}" for r in segments.itertuples()]
    return (f"Speech context: {context}\n\n"
            "Annotate every numbered sentence below. Return one entry per sentence, "
            "using the number in brackets as seg_id.\n\n" + "\n".join(lines))


def tag_with_claude(segments: pd.DataFrame, context: str, prompt_file: str = "rhetoric_v1.md") -> SpeechAnnotation:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=(PROMPTS / prompt_file).read_text(),
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": build_user_message(segments, context)}],
        output_format=SpeechAnnotation,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"model declined: {response.stop_details}")
    result = response.parsed_output
    missing = set(segments.seg_id) - {s.seg_id for s in result.sentences}
    if missing:
        raise RuntimeError(f"no annotation for seg_ids {sorted(missing)}")
    return result


def to_table(ann: SpeechAnnotation) -> pd.DataFrame:
    df = pd.DataFrame([s.model_dump() for s in ann.sentences])
    df["devices"] = df["devices"].map(lambda d: "|".join(d))
    return df.sort_values("seg_id").reset_index(drop=True)
