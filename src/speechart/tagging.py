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


def tag_with_claude(segments: pd.DataFrame, context: str, prompt_file: str = "rhetoric_v1.md",
                    log: list | None = None) -> SpeechAnnotation:
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
    if log is not None:
        log.append({"time": _now(), "model": MODEL, "status": response.stop_reason,
                    "request_id": response._request_id,
                    "raw_response": "".join(b.text for b in response.content if b.type == "text")})
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


GEMINI_MODEL = "gemini-2.5-flash"  # 3.8-flash was too often overloaded (503) in Nov 2026


def _gemini_key() -> str:
    """GEMINI_API_KEY from the environment, else from the gitignored .env file."""
    import os

    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("Set GEMINI_API_KEY (environment or AI-Art/.env)")


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tag_with_gemini(segments: pd.DataFrame, context: str, prompt_file: str = "rhetoric_v1.md",
                    model: str = GEMINI_MODEL, log: list | None = None) -> SpeechAnnotation:
    """Same prompt and schema as tag_with_claude, run on Gemini with JSON output.

    Retries on 503 overload (with backoff) and on replies that fail schema
    validation (older models follow the schema loosely). Every attempt, its
    error and the raw reply text are appended to `log` when given.
    """
    import os
    import time

    import certifi
    from google import genai
    from google.genai import errors, types

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    log = log if log is not None else []
    # v1beta: older models (e.g. 2.5-flash) reject system instructions on v1
    client = genai.Client(api_key=_gemini_key(), http_options=types.HttpOptions(api_version="v1beta"))
    config = types.GenerateContentConfig(
        system_instruction=(PROMPTS / prompt_file).read_text(),
        response_mime_type="application/json",
        response_json_schema=SpeechAnnotation.model_json_schema(),
    )
    overloads = invalid = 0
    while True:
        entry = {"time": _now(), "model": model}
        try:
            response = client.models.generate_content(
                model=model, contents=build_user_message(segments, context), config=config)
        except errors.ServerError as e:  # 503 "high demand" is common and temporary
            log.append({**entry, "status": "server_error", "error": f"{e.code} {e.status}"})
            overloads += 1
            if overloads > 5:
                raise
            wait = 15 * 2 ** (overloads - 1)
            print(f"{model}: {e.code}, retrying in {wait} s")
            time.sleep(wait)
            continue
        try:
            result = SpeechAnnotation.model_validate_json(response.text)  # enforce enums and 0-3 range
        except ValueError as e:
            log.append({**entry, "status": "invalid_output", "error": str(e), "raw_response": response.text})
            invalid += 1
            if invalid > 2:
                raise
            print(f"{model}: output failed validation ({str(e).splitlines()[0]}), retrying")
            continue
        missing = set(segments.seg_id) - {s.seg_id for s in result.sentences}
        if missing:
            log.append({**entry, "status": "incomplete", "error": f"missing seg_ids {sorted(missing)}",
                        "raw_response": response.text})
            raise RuntimeError(f"no annotation for seg_ids {sorted(missing)}")
        log.append({**entry, "status": "ok", "raw_response": response.text})
        return result
