"""Join every signal for one speech into the shared timeline.

Usage:
    uv run python scripts/build_timeline.py thunberg_2019_un

Needs (in output/<id>/): frames.csv, pauses.csv, words.csv,
reaction_scores.csv, reaction_events.csv; optional
data/speeches/<id>/annotations.json (sentence tags, see tag_sentences.py).

Writes:
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
