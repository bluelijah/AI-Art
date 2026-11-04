"""Pull the audio track out of a video (or any media file) into the speech folder.

Usage:
    uv run python scripts/import_audio.py <speech_id> <path/to/media>

Writes data/speeches/<speech_id>/audio.wav: mono, original sample rate,
16-bit. Uses PyAV, so no system ffmpeg is needed.
"""

import sys
from pathlib import Path

import av
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main(speech_id: str, media: str) -> None:
    out = ROOT / "data" / "speeches" / speech_id / "audio.wav"
    with av.open(media) as container:
        stream = container.streams.audio[0]
        sr = stream.rate
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
        chunks = []
        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):  # flush
            chunks.append(f.to_ndarray().reshape(-1))
    y = np.concatenate(chunks)
    sf.write(out, y, sr, subtype="PCM_16")
    print(f"wrote {out.relative_to(ROOT)}: {len(y) / sr:.1f} s at {sr} Hz")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
