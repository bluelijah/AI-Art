"""Audience reactions: applause, laughter, cheering.

Uses PANNs (Cnn14, trained on AudioSet's 527 sound classes). The model
tags a whole clip, so we slide a short window along the speech and read
the class probabilities for each window. That gives a time series per
class on the same axis as everything else.

Outputs:
- reaction_scores : time, speech, applause, laughter, cheering, crowd
                    (probabilities 0-1, one row per hop)
- reaction_events : label, start, end, duration, peak
                    (contiguous stretches where a reaction class is high)
"""

from __future__ import annotations

import os
import shutil
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

PANNS_DIR = Path.home() / "panns_data"
CHECKPOINT = PANNS_DIR / "Cnn14_mAP=0.431.pth"
LABELS_CSV = PANNS_DIR / "class_labels_indices.csv"
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
PANNS_SR = 32_000

# Our reaction columns -> AudioSet class names (max over the group)
CLASS_GROUPS = {
    "speech": ["Speech", "Female speech, woman speaking", "Male speech, man speaking"],
    "applause": ["Applause", "Clapping"],
    "laughter": ["Laughter", "Giggle", "Chuckle, chortle", "Belly laugh"],
    "cheering": ["Cheering"],
    "crowd": ["Crowd", "Hubbub, speech noise, speech babble"],
}
REACTIONS = ["applause", "laughter", "cheering"]


@dataclass(frozen=True)
class ReactionConfig:
    window_s: float = 2.0      # PANNs needs ~1 s or more of context
    hop_s: float = 0.25
    threshold: float = 0.20    # probability above which a reaction "is on"
    min_event_s: float = 0.75  # ignore blips shorter than this
    merge_gap_s: float = 0.5   # join events separated by less than this
    batch_size: int = 32


def ensure_panns_files() -> None:
    """Download the checkpoint (~330 MB) and label list once.

    panns_inference tries to do this with `wget`, which macOS lacks, so we
    fetch them ourselves first.
    """
    import certifi

    ctx = ssl.create_default_context(cafile=certifi.where())
    PANNS_DIR.mkdir(exist_ok=True)
    for url, dest, min_size in [(LABELS_URL, LABELS_CSV, 1_000), (CHECKPOINT_URL, CHECKPOINT, 3e8)]:
        if dest.exists() and dest.stat().st_size >= min_size:
            continue
        print(f"downloading {dest.name} ...")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, context=ctx) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        os.replace(tmp, dest)


def reaction_scores(audio_path: str | Path, cfg: ReactionConfig = ReactionConfig()) -> pd.DataFrame:
    ensure_panns_files()
    import torch
    from panns_inference import AudioTagging, labels  # import after files exist

    y, _ = librosa.load(str(audio_path), sr=PANNS_SR, mono=True)
    win, hop = int(cfg.window_s * PANNS_SR), int(cfg.hop_s * PANNS_SR)
    y = np.pad(y, (win // 2, win // 2))  # so window centres cover the whole file
    starts = np.arange(0, len(y) - win + 1, hop)

    tagger = AudioTagging(checkpoint_path=str(CHECKPOINT), device="cpu")
    idx = {name: [labels.index(c) for c in classes] for name, classes in CLASS_GROUPS.items()}

    probs = []
    with torch.inference_mode():
        for i in range(0, len(starts), cfg.batch_size):
            batch = np.stack([y[s:s + win] for s in starts[i:i + cfg.batch_size]])
            clipwise, _ = tagger.inference(batch)
            probs.append(clipwise)
    probs = np.concatenate(probs)

    df = pd.DataFrame({"time": starts / PANNS_SR})  # = window centre in the unpadded file
    for name, cols in idx.items():
        df[name] = probs[:, cols].max(axis=1)
    return df


def reaction_events(scores: pd.DataFrame, cfg: ReactionConfig = ReactionConfig()) -> pd.DataFrame:
    """Turn per-window probabilities into labelled intervals."""
    rows = []
    t = scores["time"].to_numpy()
    for label in REACTIONS:
        on = scores[label].to_numpy() >= cfg.threshold
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
        spans = [[t[a] - cfg.hop_s / 2, t[b - 1] + cfg.hop_s / 2, a, b] for a, b in zip(edges[::2], edges[1::2])]
        merged: list[list] = []
        for s in spans:
            if merged and s[0] - merged[-1][1] < cfg.merge_gap_s:
                merged[-1][1], merged[-1][3] = s[1], s[3]
            else:
                merged.append(s)
        for start, end, a, b in merged:
            if end - start >= cfg.min_event_s:
                rows.append((label, max(start, 0.0), end, end - max(start, 0.0),
                             float(scores[label].iloc[a:b].max())))
    ev = pd.DataFrame(rows, columns=["label", "start", "end", "duration", "peak"])
    return ev.sort_values("start").reset_index(drop=True)
