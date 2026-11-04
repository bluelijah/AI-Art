"""Render the artwork sketches for one speech.

Usage:
    uv run python scripts/draw.py thunberg_2019_un

Needs build_timeline.py to have run. Writes to output/<id>/art/.
First run downloads a small sentence-embedding model (~90 MB).
"""

import json
import sys
from pathlib import Path

import pandas as pd

from speechart.art import breath_scroll, fingerprint

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


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "thunberg_2019_un")
