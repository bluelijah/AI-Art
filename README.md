# Hidden Structures in Speech

We treat a speech as a set of signals running in parallel — how it is spoken,
what is said, and how people respond — put them all on one shared timeline,
and turn the patterns where they line up into artworks.

First case study: **Greta Thunberg, UN Climate Action Summit, 23 September 2019.**

## Layout

```
data/speeches/<id>/     meta.json + the recording (audio is gitignored)
src/speechart/
  delivery.py           audio -> loudness, pitch, pauses (10 ms grid)
  timeline.py           shared timeline: frame grid, word table, lead/lag
  transcribe.py         audio -> word-level transcript (WhisperX)
scripts/                one runnable script per pipeline step
tests/                  synthetic-audio checks
output/<id>/            generated tables and figures (gitignored)
```

## Setup

```bash
uv sync
uv run pytest
```

## Running

Put the recording at `data/speeches/thunberg_2019_un/audio.wav`, then:

```bash
uv run python scripts/import_audio.py thunberg_2019_un path/to/video.mp4   # once
uv run python scripts/extract_delivery.py thunberg_2019_un                 # ~1 min
uv run python scripts/transcribe.py thunberg_2019_un                       # ~6 min on an M2 CPU
```

The first transcription run downloads the Whisper and alignment models (~2 GB).

## Pipeline status

| Step | Module | Status |
|---|---|---|
| Delivery: loudness, pitch, pauses | `delivery.py` | done |
| Shared timeline + word table + lead/lag | `timeline.py` | done |
| Word-level transcript (WhisperX) | `transcribe.py` | done |
| Applause / laughter (PANNs) | | next |
| Rhetorical devices + emotion (LLM tagging) | | |
| Aftermath (GDELT, Google Trends) | | |
| Sketches: Breath Scroll, Speech Fingerprint | | |
