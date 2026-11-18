# Hidden Structures in Speech

We treat a speech as a set of signals running in parallel — how it is spoken,
what is said, and how people respond — put them all on one shared timeline,
and turn the patterns where they line up into artworks.

First case study: **Greta Thunberg, UN Climate Action Summit, 23 September 2019.**

## Layout

```
data/speeches/<id>/       meta.json, annotations.json (sentence tags), recording (gitignored: size + copyright)
prompts/                  LLM annotation prompts
src/speechart/
  delivery.py             audio -> loudness, pitch, pauses (10 ms grid)
  transcribe.py           audio -> word-level transcript (WhisperX)
  reactions.py            audio -> applause / laughter / cheering (PANNs)
  tagging.py              sentences -> rhetorical devices, emotion (Claude API)
  timeline.py             shared timeline: frame grid, word table, lead/lag
  aftermath.py            Wikipedia pageviews, GDELT news volume
  art/breath_scroll.py    ink scroll: strokes = loudness/pitch, gold = applause
  art/fingerprint.py      self-similarity matrix (what is said / how it is said)
  art/landscape.py        mountain panorama: peaks = applause, water = pauses (+ depth map)
  art/ripples.py          speech at the centre, daily echo rings (Wikipedia, GDELT)
  art/charts.py           analysis charts: phrase repetitions, anatomy of applause
scripts/                  one runnable script per pipeline step
tests/                    synthetic checks (no models, no network)
output/<id>/              generated tables and art (committed, so results are shared)
```

## Setup

```bash
uv sync
uv run pytest
```

## Running

One command per speech (after importing its audio):

```bash
uv run python scripts/import_audio.py <speech_id> path/to/video.mp4   # once: video -> audio.wav
uv run python scripts/run_pipeline.py <speech_id>                     # every step below, in order
uv run python scripts/compare.py <id> <id> ...                         # cross-speech tables and charts
```

Or step by step:

```bash
uv run python scripts/import_audio.py thunberg_2019_un path/to/video.mp4   # once: video -> audio.wav
uv run python scripts/extract_delivery.py thunberg_2019_un                 # ~1 min
uv run python scripts/transcribe.py thunberg_2019_un                       # ~6 min on an M2 CPU
uv run python scripts/detect_reactions.py thunberg_2019_un                 # ~1 min
uv run python scripts/tag_sentences.py thunberg_2019_un --provider gemini --force  # or Claude; needs a key
uv run python scripts/build_timeline.py thunberg_2019_un                   # joins everything
uv run python scripts/fetch_aftermath.py thunberg_2019_un                  # network
uv run python scripts/draw.py thunberg_2019_un                             # art -> output/<id>/art/
```

First runs download models (~2.5 GB in total: Whisper, the alignment model,
PANNs, a small sentence-embedding model).

`tag_sentences.py` needs `ANTHROPIC_API_KEY` (Claude) or `GEMINI_API_KEY`
(Gemini, in the environment or a gitignored `.env`). `annotations.json` is the
blind run (gemini-2.5-flash, transcript only, same model for every speech). `annotations_manual.json` is an
earlier manual pass made after applause was known, kept for comparison.

## Summary vs. process

Every result is kept at two levels: the summary (charts, `metrics.csv`) and
the process behind it, one row per item, so any claim can be traced back to
the sentences and moments it came from.

| Process record | Where | What it shows |
|---|---|---|
| Sentence table | `output/<id>/sentences.csv` | each sentence: text, AI tags, reaction, and how its ending was delivered |
| Run-up curves | `output/<id>/runup_curves.csv` | loudness / pitch around every sentence ending |
| Reaction moments | `output/<id>/reaction_moments.csv` | each reaction with the word and sentence before it, prediction, delivery |
| Speaker reference | `output/<id>/speaker_levels.json` | the typical loudness / pitch everything is measured against |
| LLM call log | `data/speeches/<id>/tagging_runs/*.json` | each tagging run: prompt hash, exact message sent, every attempt and raw reply |
| Pipeline log | `output/<id>/pipeline_log.jsonl` | each run: git commit, all settings, every step's duration and status |
| Cross-speech rows | `output/compare/*_all.csv` | the per-item tables of every speech stacked, behind every comparison |

## Outputs per speech

| File | One row per | What |
|---|---|---|
| `frames.csv` | 10 ms | loudness, pitch, voicing, silence |
| `pauses.csv` | pause | start, end, duration |
| `segments.csv`, `words.csv` | sentence / word | transcript with aligned times |
| `reaction_scores.csv` | 250 ms | probability of speech, applause, laughter, cheering, crowd |
| `reaction_events.csv` | reaction | label, start, end, peak |
| `sentences.csv` | sentence (main speaker) | text, rhetorical devices, emotion, intensity, addressee, topic, applause prediction, reaction that followed, ending delivery (loudness, pitch fall, pace, pause) |
| `runup_curves.csv` | sentence × 50 ms | loudness / pitch from 6 s before to 2 s after each sentence ending |
| `reaction_moments.csv` | reaction | the word and sentence before it, mid-sentence or not, tags, prediction, delivery |
| `speaker_levels.json` | | typical loudness and pitch of the main speaker |
| `pipeline_log.jsonl` | pipeline run / step | commit, settings, timings, status |
| `words_timeline.csv` | word (main speaker) | delivery + reaction that followed + sentence tags |
| `timeline_100ms.csv` | 100 ms | every continuous signal on one grid, for lead/lag |
| `aftermath_*.csv` | day | Wikipedia pageviews, GDELT news volume, echo summary |
| `art/*.png` | | Breath Scroll, Speech Fingerprint, Applause Landscape (+ depth map), Ripples, phrase repetitions, anatomy of applause |

## Pipeline status

| Step | Status |
|---|---|
| Delivery: loudness, pitch, pauses | done |
| Word-level transcript | done |
| Applause / laughter | done |
| Shared timeline | done |
| Rhetorical devices + emotion | done (blind Gemini run) |
| Aftermath | Wikipedia + GDELT done; Google Trends not yet |
| Breath Scroll, Fingerprint, Applause Landscape, Ripples | first versions |
| Gallery of Moments | not started |
| More speeches + cross-comparison | done: Thunberg 2019, Thunberg COP24 2018, Malala 2013, King 1963 (edited excerpt) |
| Process records (per-item tables, LLM and pipeline logs) | done |
