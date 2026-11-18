"""Fetch aftermath data for one speech (window and sources set in meta.json).

Usage:
    uv run python scripts/fetch_aftermath.py thunberg_2019_un

Writes output/<id>/aftermath_wikipedia.csv, aftermath_echo.csv and, when
GDELT answers, aftermath_gdelt.csv.
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from speechart.aftermath import echo_summary, gdelt_volume, wikipedia_pageviews

ROOT = Path(__file__).resolve().parents[1]


def main(speech_id: str) -> None:
    meta = json.loads((ROOT / "data" / "speeches" / speech_id / "meta.json").read_text())
    if "aftermath" not in meta:
        sys.exit(f"{speech_id}: no aftermath sources configured. {meta.get('aftermath_note', '')}")
    cfg = meta["aftermath"]
    start, end = (date.fromisoformat(d) for d in cfg["window"])
    out = ROOT / "output" / speech_id

    views = wikipedia_pageviews(cfg["wikipedia"], start, end)
    views.to_csv(out / "aftermath_wikipedia.csv", index=False)
    echo = echo_summary(views, date.fromisoformat(meta["date"]))
    echo.to_csv(out / "aftermath_echo.csv", index=False)
    print("Wikipedia echo (peak vs. median of the 14 days before):")
    print(echo.round(1).to_string(index=False))

    gpath = out / "aftermath_gdelt.csv"
    previous = pd.read_csv(gpath) if gpath.exists() else pd.DataFrame(columns=["date", "query", "articles"])
    frames = []
    for q in cfg.get("gdelt_queries", []):
        try:
            frames.append(gdelt_volume(q, start, end))
        except Exception as e:  # GDELT rate-limits hard; keep earlier results for this query
            old = previous[previous["query"] == q]
            note = f", kept {len(old)} earlier rows" if len(old) else ""
            print(f"GDELT skipped for {q!r}: {e}{note}")
            if len(old):
                frames.append(old)
    if frames:
        pd.concat(frames).to_csv(gpath, index=False)
        print(f"wrote aftermath_gdelt.csv ({len(frames)} queries)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "thunberg_2019_un")
