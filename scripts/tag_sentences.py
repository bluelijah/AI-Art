"""Tag each sentence with rhetorical devices and emotion (Claude or Gemini).

Usage:
    uv run python scripts/tag_sentences.py thunberg_2019_un [--provider gemini [--model NAME]] [--force]

Claude needs ANTHROPIC_API_KEY (or `ant auth login`); Gemini needs
GEMINI_API_KEY in the environment or in AI-Art/.env (gitignored).
Reads output/<id>/segments.csv (sentences before meta.speaker_start_s are
skipped) and writes data/speeches/<id>/annotations.json, which is kept in
git because it is costly to regenerate and is part of the research record.
Every run (successful or not) also writes data/speeches/<id>/tagging_runs/
<time>_<model>.json: the prompt hash, the exact message sent, and every
attempt with its error or raw reply.
"""

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from speechart.tagging import GEMINI_MODEL, MODEL, PROMPTS, build_user_message, tag_with_claude, tag_with_gemini

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("speech_id", nargs="?", default="thunberg_2019_un")
    ap.add_argument("--force", action="store_true", help="overwrite existing annotations")
    ap.add_argument("--provider", choices=["claude", "gemini"], default="claude")
    ap.add_argument("--model", help=f"Gemini model (default {GEMINI_MODEL})", default=GEMINI_MODEL)
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
    prompt_file = "rhetoric_v1.md"
    model = args.model if args.provider == "gemini" else MODEL

    # process record: everything sent and everything received, kept even if the run fails
    prompt_text = (PROMPTS / prompt_file).read_text()
    run = {"speech_id": args.speech_id, "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "provider": args.provider, "model": model, "prompt_file": f"prompts/{prompt_file}",
           "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
           "user_message": build_user_message(seg, context), "attempts": []}
    runs_dir = speech_dir / "tagging_runs"
    runs_dir.mkdir(exist_ok=True)
    run_path = runs_dir / f"{run['started'][:19].replace(':', '')}_{model}.json"
    try:
        if args.provider == "gemini":
            ann = tag_with_gemini(seg, context, prompt_file, model=model, log=run["attempts"])
        else:
            ann = tag_with_claude(seg, context, prompt_file, log=run["attempts"])
        run["status"] = "ok"
    except Exception as e:
        run["status"] = f"failed: {type(e).__name__}: {str(e)[:300]}"
        raise
    finally:
        run["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run_path.write_text(json.dumps(run, indent=2) + "\n")
        print(f"logged {run_path.relative_to(ROOT)} ({len(run['attempts'])} attempts, {run.get('status')})")
    source = f"{model} via {'Gemini' if args.provider == 'gemini' else 'Claude'} API (blind: transcript only)"

    dest.write_text(json.dumps({"source": source, "prompt": f"prompts/{prompt_file}", "run_log": str(run_path.relative_to(ROOT)),
                                "date": date.today().isoformat(), **ann.model_dump()}, indent=2) + "\n")
    print(f"wrote {dest.relative_to(ROOT)} ({len(ann.sentences)} sentences)")


if __name__ == "__main__":
    main()
