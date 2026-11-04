"""Tag each sentence with rhetorical devices and emotion (Claude API).

Usage:
    uv run python scripts/tag_sentences.py thunberg_2019_un [--force]

Needs Anthropic credentials (ANTHROPIC_API_KEY, or `ant auth login`).
Reads output/<id>/segments.csv (sentences before meta.speaker_start_s are
skipped) and writes data/speeches/<id>/annotations.json, which is kept in
git because it is costly to regenerate and is part of the research record.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from speechart.tagging import MODEL, SpeechAnnotation, tag_with_claude

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("speech_id", nargs="?", default="thunberg_2019_un")
    ap.add_argument("--force", action="store_true", help="overwrite existing annotations")
    args = ap.parse_args()

    speech_dir = ROOT / "data" / "speeches" / args.speech_id
    meta = json.loads((speech_dir / "meta.json").read_text())
    dest = speech_dir / "annotations.json"
    if dest.exists() and not args.force:
        raise SystemExit(f"{dest.relative_to(ROOT)} exists (source: "
                         f"{json.loads(dest.read_text())['source']}). Use --force to replace it.")

    seg = pd.read_csv(ROOT / "output" / args.speech_id / "segments.csv")
    seg = seg[seg.start >= meta.get("speaker_start_s", 0)]
    context = f"{meta['speaker']}, {meta['title']}, {meta['venue']}, {meta['date']}."
    ann = tag_with_claude(seg, context)

    dest.write_text(json.dumps({"source": f"{MODEL} via Claude API", "prompt": "prompts/rhetoric_v1.md",
                                "date": date.today().isoformat(), **ann.model_dump()}, indent=2) + "\n")
    print(f"wrote {dest.relative_to(ROOT)} ({len(ann.sentences)} sentences)")


if __name__ == "__main__":
    main()
