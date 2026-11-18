"""Render the artwork sketches for one speech.

Usage:
    uv run python scripts/draw.py thunberg_2019_un

Needs build_timeline.py to have run. Writes to output/<id>/art/.
First run downloads a small sentence-embedding model (~90 MB).
meta.json `tracked_phrase` sets the repeated phrase to follow (skipped if null).
"""

import json
import sys
from pathlib import Path

import pandas as pd

from speechart.art import breath_scroll, charts, fingerprint, landscape, ripples

ROOT = Path(__file__).resolve().parents[1]


def main(speech_id: str) -> None:
    import certifi, os
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # model download on python.org macOS builds

    meta = json.loads((ROOT / "data" / "speeches" / speech_id / "meta.json").read_text())
    out = ROOT / "output" / speech_id
    art = out / "art"
    art.mkdir(exist_ok=True)

    frames = pd.read_csv(out / "frames.csv")
    words = pd.read_csv(out / "words_timeline.csv")
    events = pd.read_csv(out / "reaction_events.csv")
    events = events[events.end > meta.get("speaker_start_s", 0)]
    grid = pd.read_csv(out / "timeline_100ms.csv")

    breath_scroll.render(frames, words, events, art / "breath_scroll.png", art / "breath_scroll_preview.png")
    print("wrote art/breath_scroll.png, art/breath_scroll_preview.png")

    fingerprint.render(grid, words, art / "speech_fingerprint.png",
                       title=f"{meta['speaker']}, {meta['title']} ({meta['date'][:4]})")
    print("wrote art/speech_fingerprint.png")

    name = f"{meta['speaker']}, {meta['title']} ({meta['date'][:4]})"
    phrase = meta.get("tracked_phrase")
    scores = pd.read_csv(out / "reaction_scores.csv")
    if phrase and charts.find_phrase(words, phrase):
        charts.how_dare_you(frames, words, scores, events, art / "phrase_repetitions.png", phrase=phrase)
    else:
        print(f"no tracked phrase found ({phrase!r}): skipping phrase_repetitions.png")
    charts.anatomy(frames, words, art / "anatomy_of_applause.png")
    print("wrote art/phrase_repetitions.png, art/anatomy_of_applause.png")

    landscape.render(grid, words, events, art / "applause_landscape.png", art / "applause_landscape_depth.png",
                     title=f"Applause Landscape: {name}")
    print("wrote art/applause_landscape.png (+ _depth.png for depth-conditioned painting)")

    wiki_path, gdelt_path = out / "aftermath_wikipedia.csv", out / "aftermath_gdelt.csv"
    if wiki_path.exists() and "aftermath" in meta:
        gdelt = pd.read_csv(gdelt_path) if gdelt_path.exists() else None
        if gdelt is not None:
            gdelt = gdelt[gdelt["query"] == meta["aftermath"]["gdelt_queries"][0]]
        starts = [words.start[i] for i in charts.find_phrase(words, phrase)] if phrase else []
        ripples.render(frames, words, events, pd.read_csv(wiki_path), gdelt, meta["date"], art / "ripples.png",
                       meta["aftermath"]["wikipedia"][0], starts, (phrase or "").capitalize(), title=f"Ripples: {name}")
        print("wrote art/ripples.png")
    else:
        print("no aftermath data yet: run fetch_aftermath.py for ripples.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "thunberg_2019_un")
