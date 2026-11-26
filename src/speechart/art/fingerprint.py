"""Speech Fingerprint: a self-similarity matrix, borrowed from music analysis.

Time runs along both axes. Cell (i, j) is bright when moment i resembles
moment j, so repetition shows up as off-diagonal threads and sections as
blocks along the diagonal: a woven square that is different for every
speech.

Two kinds of similarity share the square:
- upper triangle  = WHAT is said: cosine similarity of sentence-embedding
                    vectors for the words around each moment
- lower triangle  = HOW it is said: similarity of loudness, pitch, pace
                    and silence around each moment
Moments with no speech (pauses, applause) stay dark in the text half.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BG = "#0e0d12"


def text_windows(words: pd.DataFrame, grid_t: np.ndarray, half_words: int = 5) -> list[str | None]:
    """For each grid time, the ~11 words centerd on the word being spoken (None if silent)."""
    out = []
    starts, ends = words.start.to_numpy(), words.end.to_numpy()
    toks = words.word.tolist()
    for t in grid_t:
        i = np.searchsorted(starts, t, side="right") - 1
        if i < 0 or t > ends[i] + 0.3:  # not inside (or just after) a word
            out.append(None)
            continue
        out.append(" ".join(toks[max(0, i - half_words): i + half_words + 1]))
    return out


def text_similarity(windows: list[str | None]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    idx = [i for i, w in enumerate(windows) if w]
    emb = model.encode([windows[i] for i in idx], normalize_embeddings=True, batch_size=64)
    n = len(windows)
    sim = np.full((n, n), np.nan)
    sim[np.ix_(idx, idx)] = emb @ emb.T
    return sim


def delivery_similarity(grid: pd.DataFrame, cols=("loudness_db", "pitch_hz", "pause", "voiced_prob"),
                        smooth_bins: int = 5) -> np.ndarray:
    X = grid[list(cols)].copy()
    X["pitch_hz"] = X["pitch_hz"].fillna(X["pitch_hz"].median())
    X = X.rolling(smooth_bins, center=True, min_periods=1).mean()
    X = (X - X.mean()) / X.std()
    d = np.sqrt(((X.to_numpy()[:, None, :] - X.to_numpy()[None, :, :]) ** 2).sum(-1))
    return np.exp(-d / np.median(d))


def render(grid100: pd.DataFrame, words: pd.DataFrame, path: str, step_s: float = 0.5,
           title: str | None = None) -> None:
    g = grid100.copy()
    g["bin"] = (g.time // step_s).astype(int)
    g = g.groupby("bin").mean(numeric_only=True).reset_index(drop=True)
    t = g.time.to_numpy()

    S_text = text_similarity(text_windows(words, t))
    S_del = delivery_similarity(g)

    n = len(t)
    upper = np.triu(np.ones((n, n), bool), 1)
    lower = np.tril(np.ones((n, n), bool), -1)

    text_cmap = LinearSegmentedColormap.from_list("t", [BG, "#2b3a67", "#7fb7d9", "#f2f7fb"])
    del_cmap = LinearSegmentedColormap.from_list("d", [BG, "#5a1f3d", "#d9776a", "#fbeee6"])

    # contrast-stretch each half so the weave is visible
    st = np.where(upper, S_text, np.nan)
    lo, hi = np.nanpercentile(st, [50, 99.5])
    st = np.clip((st - lo) / (hi - lo), 0, 1)
    sd = np.where(lower, S_del, np.nan)
    lo, hi = np.nanpercentile(sd, [40, 99.5])
    sd = np.clip((sd - lo) / (hi - lo), 0, 1)

    fig = plt.figure(figsize=(10, 10.8), dpi=150, facecolor=BG)
    ax = fig.add_axes([0.04, 0.07, 0.92, 0.855])
    ax.set_facecolor(BG)
    ext = [t[0], t[-1], t[-1], t[0]]
    ax.imshow(st, cmap=text_cmap, vmin=0, vmax=1, extent=ext, interpolation="nearest")
    ax.imshow(sd, cmap=del_cmap, vmin=0, vmax=1, extent=ext, interpolation="nearest")

    # applause along the diagonal as gold ticks
    for col in [c for c in g.columns if c == "applause_event"]:
        on = g[col].to_numpy() > 0.5
        ax.scatter(t[on], t[on], s=6, color="#c9a227", lw=0, zorder=3)

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.04, 0.955, title or "Speech Fingerprint", color="#e9e6f0", fontsize=14, family="serif")
    fig.text(0.04, 0.035, "upper: what is said (text embeddings)    lower: how it is said (loudness, pitch, silence)"
             "    gold: applause", color="#8b8798", fontsize=8, family="serif")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
