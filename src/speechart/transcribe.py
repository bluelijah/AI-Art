"""Word-level transcript: what is said, and exactly when.

WhisperX in two passes:
1. Whisper (faster-whisper backend) transcribes into segments.
2. A wav2vec2 model force-aligns each word to the audio, giving word
   start/end times precise enough to measure the pause after each word.

Audio is passed in as an array, so no system ffmpeg is needed.

Output tables (both in seconds):
- segments : seg_id, start, end, text
- words    : word, start, end, score, seg_id
  (score = alignment confidence; words the aligner could not place,
   usually numbers, get times interpolated from their neighbours)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

WHISPER_SR = 16_000


@dataclass(frozen=True)
class TranscribeConfig:
    model: str = "large-v3-turbo"   # good accuracy, ~8x faster than large-v3
    language: str = "en"
    device: str = "cpu"             # CTranslate2 has no Apple GPU backend
    compute_type: str = "int8"      # fastest on CPU
    batch_size: int = 8


def transcribe(audio_path: str | Path, cfg: TranscribeConfig = TranscribeConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    import certifi
    import whisperx  # heavy import, keep it local

    # python.org macOS builds ship without CA certificates, which breaks the
    # model downloads (torch.hub / urllib). Point them at certifi's bundle.
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())

    audio, _ = librosa.load(str(audio_path), sr=WHISPER_SR, mono=True)
    audio = audio.astype(np.float32)

    model = whisperx.load_model(cfg.model, cfg.device, compute_type=cfg.compute_type, language=cfg.language)
    result = model.transcribe(audio, batch_size=cfg.batch_size, language=cfg.language)

    align_model, meta = whisperx.load_align_model(language_code=cfg.language, device=cfg.device)
    aligned = whisperx.align(result["segments"], align_model, meta, audio, cfg.device,
                             interpolate_method="nearest")

    seg_rows, word_rows = [], []
    for i, seg in enumerate(aligned["segments"]):
        seg_rows.append((i, seg["start"], seg["end"], seg["text"].strip()))
        for w in seg.get("words", []):
            word_rows.append((w["word"], w.get("start"), w.get("end"), w.get("score"), i))

    segments = pd.DataFrame(seg_rows, columns=["seg_id", "start", "end", "text"])
    words = pd.DataFrame(word_rows, columns=["word", "start", "end", "score", "seg_id"])
    words = _fill_missing_times(words)
    return segments, words


def _fill_missing_times(words: pd.DataFrame) -> pd.DataFrame:
    """Unaligned words (e.g. '2019', '50%') come back without times.
    Place them between their neighbours so every word sits on the timeline."""
    w = words.copy()
    w["aligned"] = w["start"].notna()
    w["start"] = w["start"].astype(float).interpolate(limit_direction="both")
    w["end"] = w["end"].astype(float).interpolate(limit_direction="both")
    w["end"] = np.maximum(w["end"], w["start"])
    return w
