"""Tag each sentence with rhetorical devices and emotion (Claude or Gemini).

Usage:
    uv run python scripts/tag_sentences.py thunberg_2019_un [--provider gemini [--model NAME]] [--runs 5] [--force]

The model is run --runs times independently (default 5) and the tags are
combined by vote (see tagging.consensus). annotations.json holds the
consensus plus a per-sentence agreement record (applause_vote, emotion and
device agreement), so every tag comes with how sure the runs were.

Claude needs ANTHROPIC_API_KEY (or `ant auth login`); Gemini needs
GEMINI_API_KEY in the environment or in AI-Art/.env (gitignored).
Reads output/<id>/segments.csv (sentences before meta.speaker_start_s are
skipped) and writes data/speeches/<id>/annotations.json, which is kept in
git because it is costly to regenerate and is part of the research record.
Every invocation (successful or not) also writes data/speeches/<id>/
tagging_runs/<time>_<model>_x<N>.json: the prompt hash, the exact message
sent, and for each run every attempt with its error or raw reply.
"""

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from speechart.tagging import (GEMINI_MODEL, MODEL, PROMPTS, build_user_message, consensus,
                               tag_with_claude, tag_with_gemini)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("speech_id", nargs="?", default="thunberg_2019_un")
    ap.add_argument("--force", action="store_true", help="overwrite existing annotations")
    ap.add_argument("--provider", choices=["claude", "gemini"], default="claude")
    ap.add_argument("--model", help=f"Gemini model (default {GEMINI_MODEL})", default=GEMINI_MODEL)
    ap.add_argument("--runs", type=int, default=5, help="independent runs to combine by vote (default 5)")
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

    # process record: everything sent and everything received, kept even if runs fail
    prompt_text = (PROMPTS / prompt_file).read_text()
    log = {"speech_id": args.speech_id, "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "provider": args.provider, "model": model, "n_runs_requested": args.runs,
           "prompt_file": f"prompts/{prompt_file}",
           "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
           "user_message": build_user_message(seg, context), "runs": []}
    runs_dir = speech_dir / "tagging_runs"
    runs_dir.mkdir(exist_ok=True)
    log_path = runs_dir / f"{log['started'][:19].replace(':', '')}_{model}_x{args.runs}.json"

    results = []
    try:
        for i in range(args.runs):
            entry = {"run": i + 1, "attempts": []}
            log["runs"].append(entry)
            try:
                if args.provider == "gemini":
                    results.append(tag_with_gemini(seg, context, prompt_file, model=model, log=entry["attempts"]))
                else:
                    results.append(tag_with_claude(seg, context, prompt_file, log=entry["attempts"]))
                entry["status"] = "ok"
            except Exception as e:  # one bad run shouldn't sink the rest
                entry["status"] = f"failed: {type(e).__name__}: {str(e)[:300]}"
            print(f"run {i + 1}/{args.runs}: {entry['status']}", flush=True)
    finally:
        log["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log["n_runs_ok"] = len(results)
        log_path.write_text(json.dumps(log, indent=2) + "\n")
        print(f"logged {log_path.relative_to(ROOT)} ({len(results)}/{args.runs} runs ok)")

    if len(results) < max(1, (args.runs + 1) // 2):
        raise SystemExit(f"only {len(results)}/{args.runs} runs succeeded: not enough for a consensus")
    ann, agreement = consensus(results)

    source = (f"{model} via {'Gemini' if args.provider == 'gemini' else 'Claude'} API, consensus of "
              f"{len(results)} independent runs (blind: transcript only)")
    dest.write_text(json.dumps({
        "source": source, "prompt": f"prompts/{prompt_file}", "run_log": str(log_path.relative_to(ROOT)),
        "date": date.today().isoformat(), "n_runs": len(results), **ann.model_dump(),
        "agreement": agreement.round(3).to_dict(orient="records"),
    }, indent=2) + "\n")
    print(f"wrote {dest.relative_to(ROOT)} ({len(ann.sentences)} sentences, "
          f"mean applause-vote agreement {agreement.applause_vote.apply(lambda v: max(v, 1 - v)).mean():.0%})")


if __name__ == "__main__":
    main()
