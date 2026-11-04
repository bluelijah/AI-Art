"""Delivery features: how the speech is spoken.

Everything here works on the raw audio alone (no transcript needed):

- loudness   : RMS energy per frame, in dB
- pitch      : fundamental frequency per frame (pYIN), NaN when unvoiced
- pauses     : stretches where the level drops to the room's noise floor

All frame-level features share one hop size, so they line up row-for-row
and can be dropped straight onto the shared timeline (see timeline.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DeliveryConfig:
    sr: int = 16_000            # resample everything to this rate
    hop_s: float = 0.010        # 10 ms frames: fine enough for pause edges
    frame_s: float = 0.040      # analysis window for loudness / pitch
    fmin_hz: float = 75.0       # pitch search range (covers most adult voices)
    fmax_hz: float = 500.0
    min_pause_s: float = 0.20   # shorter gaps are just articulation
    # Silence threshold sits this fraction of the way from the noise floor
    # (10th percentile of level) up to typical speech (90th percentile).
    silence_ratio: float = 0.30

    @property
    def hop_length(self) -> int:
        return int(round(self.hop_s * self.sr))

    @property
    def frame_length(self) -> int:
        return int(round(self.frame_s * self.sr))


def load_audio(path: str | Path, cfg: DeliveryConfig = DeliveryConfig()) -> np.ndarray:
    """Load any audio file as mono float32 at cfg.sr."""
    y, _ = librosa.load(str(path), sr=cfg.sr, mono=True)
    return y


def frame_features(y: np.ndarray, cfg: DeliveryConfig = DeliveryConfig()) -> pd.DataFrame:
    """Per-frame loudness and pitch.

    Returns one row per hop with columns:
    time, loudness_db, pitch_hz, voiced_prob, is_silent
    """
    hop, win = cfg.hop_length, cfg.frame_length

    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
    loudness_db = librosa.amplitude_to_db(rms, ref=1.0, top_db=None)

    f0, _voiced, voiced_prob = librosa.pyin(
        y,
        fmin=cfg.fmin_hz,
        fmax=cfg.fmax_hz,
        sr=cfg.sr,
        frame_length=1024,  # 64 ms: pYIN needs room for low pitches
        hop_length=hop,
        center=True,
    )

    n = min(len(loudness_db), len(f0))
    times = librosa.frames_to_time(np.arange(n), sr=cfg.sr, hop_length=hop)

    df = pd.DataFrame(
        {
            "time": times,
            "loudness_db": loudness_db[:n],
            "pitch_hz": f0[:n],
            "voiced_prob": voiced_prob[:n],
        }
    )
    df["is_silent"] = df["loudness_db"] < silence_threshold_db(df["loudness_db"], cfg)
    return df


def silence_threshold_db(loudness_db: pd.Series | np.ndarray, cfg: DeliveryConfig) -> float:
    """Adaptive threshold, so it works for quiet and loud recordings alike."""
    floor, speech = np.percentile(loudness_db, [10, 90])
    return float(floor + cfg.silence_ratio * (speech - floor))


def detect_pauses(frames: pd.DataFrame, cfg: DeliveryConfig = DeliveryConfig()) -> pd.DataFrame:
    """Group silent frames into pause intervals.

    A frame counts as silent only when its whole analysis window is quiet,
    so each silent run is widened by half a window on both sides to recover
    the true gap.

    Returns columns: start, end, duration. Pauses touching the very start
    or end of the file are dropped (that is lead-in, not a pause).

    Note: applause is loud, so it is *not* a pause here. The reactions
    module labels it separately; the pause after a phrase that is filled
    by applause shows up there, not here.
    """
    silent = frames["is_silent"].to_numpy()
    times = frames["time"].to_numpy()
    if len(silent) == 0:
        return pd.DataFrame(columns=["start", "end", "duration"])

    # run-length encode the boolean mask
    edges = np.flatnonzero(np.diff(silent.astype(np.int8))) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(silent)]

    rows = []
    for s, e in zip(starts, ends):
        if not silent[s]:
            continue
        if s == 0 or e == len(silent):
            continue  # lead-in / tail
        half = cfg.frame_s / 2
        start, end = times[s] - half, times[e - 1] + half
        if end - start >= cfg.min_pause_s:
            rows.append((start, end, end - start))

    return pd.DataFrame(rows, columns=["start", "end", "duration"])


def extract_delivery(path: str | Path, cfg: DeliveryConfig = DeliveryConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: audio file -> (frames, pauses)."""
    y = load_audio(path, cfg)
    frames = frame_features(y, cfg)
    return frames, detect_pauses(frames, cfg)
