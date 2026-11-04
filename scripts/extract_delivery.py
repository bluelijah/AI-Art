"""Run delivery-feature extraction for one speech.

Usage:
    uv run python scripts/extract_delivery.py thunberg_2019_un

Reads  data/speeches/<id>/meta.json and the audio file it names.
Writes output/<id>/frames.csv (10 ms grid) and output/<id>/pauses.csv.
"""

import json
import sys
from pathlib import Path

from speechart.delivery import extract_delivery

ROOT = Path(__file__).resolve().parents[1]


def main(speech_id: str) -> None:
    speech_dir = ROOT / "data" / "speeches" / speech_id
    meta = json.loads((speech_dir / "meta.json").read_text())
    audio = speech_dir / meta["audio_file"]
    if not audio.exists():
        sys.exit(f"No audio at {audio}. Add the recording first (see meta.json notes).")

    frames, pauses = extract_delivery(audio)

    out = ROOT / "output" / speech_id
    out.mkdir(parents=True, exist_ok=True)
    frames.to_csv(out / "frames.csv", index=False)
    pauses.to_csv(out / "pauses.csv", index=False)

    dur = frames["time"].iloc[-1]
    print(f"{speech_id}: {dur:.1f} s of audio, {len(pauses)} pauses "
          f"(median {pauses['duration'].median():.2f} s, longest {pauses['duration'].max():.2f} s)")
    print(f"wrote {out.relative_to(ROOT)}/frames.csv, pauses.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "thunberg_2019_un")
