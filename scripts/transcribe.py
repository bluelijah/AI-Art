"""Word-level transcript for one speech, joined with delivery features.

Usage:
    uv run python scripts/transcribe.py thunberg_2019_un [--model large-v3-turbo]

Writes to output/<id>/:
  segments.csv        Whisper segments (roughly sentences)
  words.csv           one row per word with aligned start/end
  words_delivery.csv  words + loudness, pitch and pause_after
                      (needs frames.csv from extract_delivery.py)
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from speechart.timeline import words_with_delivery
from speechart.transcribe import TranscribeConfig, transcribe

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("speech_id", nargs="?", default="thunberg_2019_un")
    ap.add_argument("--model", default=TranscribeConfig.model)
    args = ap.parse_args()

    speech_dir = ROOT / "data" / "speeches" / args.speech_id
    meta = json.loads((speech_dir / "meta.json").read_text())
    out = ROOT / "output" / args.speech_id
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = TranscribeConfig(model=args.model, language=meta.get("language", "en"))
    segments, words = transcribe(speech_dir / meta["audio_file"], cfg)
    segments.to_csv(out / "segments.csv", index=False)
    words.to_csv(out / "words.csv", index=False)
    print(f"{len(segments)} segments, {len(words)} words "
          f"({(~words['aligned']).sum()} unaligned, interpolated) in {time.time() - t0:.0f} s")

    frames_path = out / "frames.csv"
    if frames_path.exists():
        wd = words_with_delivery(words, pd.read_csv(frames_path))
        wd.to_csv(out / "words_delivery.csv", index=False)
        print(f"wrote words_delivery.csv")
    else:
        print("no frames.csv yet: run extract_delivery.py, then rerun to get words_delivery.csv")


if __name__ == "__main__":
    main()
