"""Applause / laughter / cheering for one speech (PANNs).

Usage:
    uv run python scripts/detect_reactions.py thunberg_2019_un

Writes output/<id>/reaction_scores.csv and reaction_events.csv.
First run downloads the PANNs checkpoint (~330 MB) to ~/panns_data.
"""

import json
import sys
from pathlib import Path

from speechart.reactions import reaction_events, reaction_scores

ROOT = Path(__file__).resolve().parents[1]


def main(speech_id: str) -> None:
    speech_dir = ROOT / "data" / "speeches" / speech_id
    meta = json.loads((speech_dir / "meta.json").read_text())
    out = ROOT / "output" / speech_id
    out.mkdir(parents=True, exist_ok=True)

    scores = reaction_scores(speech_dir / meta["audio_file"])
    events = reaction_events(scores)
    scores.to_csv(out / "reaction_scores.csv", index=False)
    events.to_csv(out / "reaction_events.csv", index=False)
    print(events.round(2).to_string(index=False) if len(events) else "no reaction events found")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "thunberg_2019_un")
