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

Every run appends to output/<id>/pipeline_log.jsonl: the git commit, all
step settings (thresholds, models), and each step's command, duration
and status.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["delivery", "transcribe", "reactions", "tag", "timeline", "aftermath", "draw"]


LOG: Path | None = None


def settings() -> dict:
    """The parameters every step ran with, so results can be traced to their settings."""
    from dataclasses import asdict

    from speechart.delivery import DeliveryConfig
    from speechart.reactions import ReactionConfig
    from speechart.tagging import GEMINI_MODEL, MODEL
    from speechart.transcribe import TranscribeConfig

    return {"delivery": asdict(DeliveryConfig()), "transcribe": asdict(TranscribeConfig()),
            "reactions": asdict(ReactionConfig()), "tagging": {"gemini": GEMINI_MODEL, "claude": MODEL}}


def log(entry: dict) -> None:
    if LOG is not None:
        with LOG.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(step: str, *args: str) -> None:
    print(f"\n=== {' '.join(args)}", flush=True)
    start, t = now(), time.time()
    status = "ok"
    try:
        subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as e:
        status = f"failed (exit {e.returncode})"
        raise
    finally:
        log({"step": step, "command": " ".join(args), "started": start, "seconds": round(time.time() - t, 1),
             "status": status})


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

    global LOG
    LOG = ROOT / "output" / sid / "pipeline_log.jsonl"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    log({"step": "run", "started": now(), "from_step": a.start, "provider": a.provider,
         "git_commit": commit.stdout.strip(), "settings": settings(),
         "speaker_start_s": meta.get("speaker_start_s"), "tracked_phrase": meta.get("tracked_phrase")})

    for step in STEPS[STEPS.index(a.start):]:
        if step == "delivery":
            run(step, "scripts/extract_delivery.py", sid)
        elif step == "transcribe":
            run(step, "scripts/transcribe.py", sid)
        elif step == "reactions":
            run(step, "scripts/detect_reactions.py", sid)
        elif step == "tag":
            if (speech_dir / "annotations.json").exists():
                print("\n=== tag: annotations.json exists, skipping")
                log({"step": step, "status": "skipped: annotations.json exists"})
            else:
                run(step, "scripts/tag_sentences.py", sid, "--provider", a.provider)
        elif step == "timeline":
            run(step, "scripts/build_timeline.py", sid)
        elif step == "aftermath":
            if "aftermath" in meta:
                run(step, "scripts/fetch_aftermath.py", sid)
            else:
                print(f"\n=== aftermath: skipped. {meta.get('aftermath_note', 'no sources configured')}")
                log({"step": step, "status": "skipped: " + meta.get("aftermath_note", "no sources configured")})
        elif step == "draw":
            run(step, "scripts/draw.py", sid)


if __name__ == "__main__":
    main()
