# Hidden Structures in Speech

We treat a speech as a set of signals running in parallel — how it is spoken,
what is said, and how people respond — put them all on one shared timeline,
and turn the patterns where they line up into artworks.

First case study: **Greta Thunberg, UN Climate Action Summit, 23 September 2019.**

## Layout

```
data/speeches/<id>/       meta.json, annotations.json (sentence tags), recording (gitignored)
prompts/                  LLM annotation prompts
src/speechart/
  delivery.py             audio -> loudness, pitch, pauses (10 ms grid)
  transcribe.py           audio -> word-level transcript (WhisperX)
  reactions.py            audio -> applause / laughter / cheering (PANNs)
  tagging.py              sentences -> rhetorical devices, emotion (Claude API)
  timeline.py             shared timeline: frame grid, word table, lead/lag
  aftermath.py            Wikipedia pageviews, GDELT news volume
  art/breath_scroll.py    sketch: ink scroll
  art/fingerprint.py      sketch: self-similarity matrix
scripts/                  one runnable script per pipeline step
tests/                    synthetic checks (no models, no network)
output/<id>/              generated tables and art (gitignored)
```

## Setup

```bash
uv sync
uv run pytest
```

## Running

```bash
uv run python scripts/import_audio.py thunberg_2019_un path/to/video.mp4   # once: video -> audio.wav
uv run python scripts/extract_delivery.py thunberg_2019_un                 # ~1 min
uv run python scripts/transcribe.py thunberg_2019_un                       # ~6 min on an M2 CPU
uv run python scripts/detect_reactions.py thunberg_2019_un                 # ~1 min
uv run python scripts/tag_sentences.py thunberg_2019_un --force            # needs Anthropic credentials
uv run python scripts/build_timeline.py thunberg_2019_un                   # joins everything
uv run python scripts/draw.py thunberg_2019_un                             # art -> output/<id>/art/
uv run python scripts/fetch_aftermath.py thunberg_2019_un                  # network
```

First runs download models (~2.5 GB in total: Whisper, the alignment model,
PANNs, a small sentence-embedding model).

`tag_sentences.py` needs `ANTHROPIC_API_KEY` (or `ant auth login`). The
committed `annotations.json` is a manual first pass made with the same prompt
and schema; it was made *after* applause had been detected, so it is not a
blind test of `invites_applause`. Re-run the script to get a blind one.

## Outputs per speech

| File | One row per | What |
|---|---|---|
| `frames.csv` | 10 ms | loudness, pitch, voicing, silence |
| `pauses.csv` | pause | start, end, duration |
| `segments.csv`, `words.csv` | sentence / word | transcript with aligned times |
| `reaction_scores.csv` | 250 ms | probability of speech, applause, laughter, cheering, crowd |
| `reaction_events.csv` | reaction | label, start, end, peak |
| `words_timeline.csv` | word (main speaker) | delivery + reaction that followed + sentence tags |
| `timeline_100ms.csv` | 100 ms | every continuous signal on one grid, for lead/lag |
| `aftermath_*.csv` | day | Wikipedia pageviews, GDELT news volume, echo summary |
| `art/*.png` | | Breath Scroll (long + preview), Speech Fingerprint |

## Pipeline status

| Step | Status |
|---|---|
| Delivery: loudness, pitch, pauses | done |
| Word-level transcript | done |
| Applause / laughter | done |
| Shared timeline | done |
| Rhetorical devices + emotion | code done; current tags are a manual, non-blind pass |
| Aftermath | Wikipedia + GDELT done; Google Trends not yet |
| Sketches: Breath Scroll, Speech Fingerprint | first versions |
| Applause Landscape, Ripples, Gallery of Moments | not started |
