"""Run every pipeline step for one speech, in order.

Usage:
    uv run python scripts/run_pipeline.py <speech_id> [--from STEP] [--provider gemini|claude]

Expects data/speeches/<id>/audio.wav (make it with import_audio.py).
Steps: delivery, transcribe, reactions, tag, timeline, aftermath, draw.
Sentence tagging is skipped if annotations.json exists (delete it or run
tag_sentences.py --force to redo). Aftermath is skipped when meta.json
configures no sources (e.g. speeches before 2015).

After `transcribe`, check output/<id>/segments.csv: if the recording opens
with someone else (host, interviewer), set speaker_start_s in meta.json and
rerun with --from tag.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["delivery", "transcribe", "reactions", "tag", "timeline", "aftermath", "draw"]


def run(*args: str) -> None:
    print(f"\n=== {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("speech_id")
    ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    ap.add_argument("--provider", choices=["gemini", "claude"], default="gemini")
    a = ap.parse_args()

    sid = a.speech_id
    speech_dir = ROOT / "data" / "speeches" / sid
    meta = json.loads((speech_dir / "meta.json").read_text())
    if not (speech_dir / meta["audio_file"]).exists():
        sys.exit(f"no {meta['audio_file']} in {speech_dir}: run scripts/import_audio.py {sid} <video> first")

    for step in STEPS[STEPS.index(a.start):]:
        if step == "delivery":
            run("scripts/extract_delivery.py", sid)
        elif step == "transcribe":
            run("scripts/transcribe.py", sid)
        elif step == "reactions":
            run("scripts/detect_reactions.py", sid)
        elif step == "tag":
            if (speech_dir / "annotations.json").exists():
                print("\n=== tag: annotations.json exists, skipping")
            else:
                run("scripts/tag_sentences.py", sid, "--provider", a.provider)
        elif step == "timeline":
            run("scripts/build_timeline.py", sid)
        elif step == "aftermath":
            if "aftermath" in meta:
                run("scripts/fetch_aftermath.py", sid)
            else:
                print(f"\n=== aftermath: skipped. {meta.get('aftermath_note', 'no sources configured')}")
        elif step == "draw":
            run("scripts/draw.py", sid)


if __name__ == "__main__":
    main()
