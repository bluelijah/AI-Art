"""Join every signal for one speech into the shared timeline.

Usage:
    uv run python scripts/build_timeline.py thunberg_2019_un

Needs (in output/<id>/): frames.csv, pauses.csv, words.csv,
reaction_scores.csv, reaction_events.csv; optional
data/speeches/<id>/annotations.json (sentence tags, see tag_sentences.py).

Writes:
  sentences.csv         one row per sentence: text, tags (devices, emotion, ...),
                        the reaction that followed, and how its ending was
                        delivered (loudness, pitch fall, pace, pause after)
  runup_curves.csv      loudness / pitch around every sentence ending (-6..+2 s),
                        relative to the speaker's typical level: the data
                        behind the anatomy-of-applause charts
  reaction_moments.csv  every detected reaction with the word and sentence
                        before it, its tags, the AI's prediction, the delivery
                        run-up, and whether the crowd broke in mid-sentence
  speaker_levels.json   the speaker's typical loudness and pitch (the reference)
  words_timeline.csv    one row per word spoken by the main speaker:
                        delivery + reaction that followed + sentence tags
  timeline_100ms.csv    100 ms grid: loudness, pitch, pause, reactions,
                        current sentence id. Input for lead/lag analysis.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from speechart.process import ending_stats, reaction_moments, runup_curves, speaker_levels
from speechart.tagging import SpeechAnnotation, to_table
from speechart.timeline import (clip_long_words, mark_intervals, reaction_after,
                                resample, words_with_delivery)

ROOT = Path(__file__).resolve().parents[1]


def main(speech_id: str) -> None:
    meta = json.loads((ROOT / "data" / "speeches" / speech_id / "meta.json").read_text())
    out = ROOT / "output" / speech_id
    read = lambda name: pd.read_csv(out / name)
    t0 = meta.get("speaker_start_s", 0.0)

    frames, pauses = read("frames.csv"), read("pauses.csv")
    scores, events = read("reaction_scores.csv"), read("reaction_events.csv")

    words = clip_long_words(read("words.csv"))
    words = words[words["start"] >= t0].reset_index(drop=True)
    words = words_with_delivery(words, frames)
    words = reaction_after(words, events)

    ann_path = ROOT / "data" / "speeches" / speech_id / "annotations.json"
    if ann_path.exists():
        ann = to_table(SpeechAnnotation.model_validate(json.loads(ann_path.read_text())))
        words = words.merge(ann, on="seg_id", how="left")
    words.to_csv(out / "words_timeline.csv", index=False)

    # one readable row per sentence: text next to its tags and what followed it
    seg = pd.read_csv(out / "segments.csv")
    seg = seg[seg.seg_id.isin(words.seg_id.unique())][["seg_id", "text"]]
    per = words.groupby("seg_id").agg(start=("start", "min"), end=("end", "max"),
                                      reaction_after=("reaction_after", lambda x: next((r for r in x if r), "")))
    sentences = seg.merge(per, on="seg_id")
    if ann_path.exists():
        sentences = sentences.merge(ann, on="seg_id", how="left")

    # process data: how each sentence ending was delivered, and the curves behind it
    levels = speaker_levels(frames, words)
    sentences = sentences.merge(ending_stats(words, frames, levels), on="seg_id", how="left")
    sentences.round(3).to_csv(out / "sentences.csv", index=False)
    curves = runup_curves(words, frames, levels)
    curves.round(3).to_csv(out / "runup_curves.csv", index=False)
    moments = reaction_moments(events[events.end > t0], words, sentences)
    moments.round(3).to_csv(out / "reaction_moments.csv", index=False)
    (out / "speaker_levels.json").write_text(json.dumps({k: round(v, 3) for k, v in levels.items()}, indent=2) + "\n")

    # 100 ms grid
    grid = mark_intervals(frames, pauses, "pause")
    for label in events["label"].unique():
        grid = mark_intervals(grid, events[events.label == label], f"{label}_event")
    grid = resample(grid.drop(columns=["is_silent"]), 0.1)
    s = scores.copy()
    s["time"] = (s["time"] / 0.1).round() * 0.1
    grid["time"] = grid["time"].round(1)
    grid = grid.merge(s.groupby("time").mean().add_suffix("_prob").reset_index(), on="time", how="left")
    probs = [c for c in grid if c.endswith("_prob")]
    grid[probs] = grid[probs].interpolate(limit_direction="both")
    seg = np.full(len(grid), -1)
    for r in words.groupby("seg_id").agg(start=("start", "min"), end=("end", "max")).reset_index().itertuples():
        seg[(grid.time >= r.start) & (grid.time < r.end)] = r.seg_id
    grid["seg_id"] = seg
    grid = grid[grid.time >= t0].reset_index(drop=True)
    grid.to_csv(out / "timeline_100ms.csv", index=False)

    rx = words[words.reaction_after != ""]
    print(f"{len(words)} words from {t0:.1f} s ({words.clipped.sum()} stretched words fixed), grid {len(grid)} rows")
    print("reactions credited to:")
    for r in rx.itertuples():
        print(f"  {r.reaction_after:9s} {r.reaction_duration:5.1f} s after '{r.word}' ({r.end:.1f} s, delay {r.reaction_delay:.1f} s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "thunberg_2019_un")
