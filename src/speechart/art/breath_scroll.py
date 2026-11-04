"""Breath Scroll: the speech as one long ink scroll, read left to right.

Data drives every mark:
- each word is an ink stroke; its thickness follows the frame-by-frame
  loudness inside the word, and its height follows pitch
- pauses are blank paper
- applause is gold leaf spanning the time the room responded
- sentences the text model flagged as built to invite applause get a
  small red seal under their final word (prediction vs. the gold = reality)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

PAPER = "#f3ecdc"
INK = "#1b1a17"
GOLD = "#c9a227"
SEAL = "#b0302a"


def _norm(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def draw_scroll(ax, frames: pd.DataFrame, words: pd.DataFrame, events: pd.DataFrame,
                t0: float, t1: float, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    ax.set_facecolor(PAPER)
    ax.set_xlim(t0, t1)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")

    # paper grain
    gx = rng.uniform(t0, t1, 4000)
    gy = rng.uniform(-1.3, 1.3, 4000)
    ax.scatter(gx, gy, s=0.3, color="#d9cfb8", alpha=0.6, lw=0, zorder=0)

    # gold leaf for applause: layered translucent rectangles with ragged edges
    for ev in events[(events.label == "applause") & (events.end > t0) & (events.start < t1)].itertuples():
        for k in range(14):
            jitter = rng.normal(0, 0.12, 2)
            ax.add_patch(Rectangle((ev.start + jitter[0], -1.1 + rng.uniform(0, 0.15)),
                                   ev.duration + jitter[1] - jitter[0], 2.1 - rng.uniform(0, 0.2),
                                   color=GOLD, alpha=0.07, lw=0, zorder=1))
        flecks = rng.uniform(ev.start, ev.end, int(40 * ev.duration))
        ax.scatter(flecks, rng.uniform(-1.05, 1.0, len(flecks)), s=rng.uniform(1, 9, len(flecks)),
                   color="#e3c45a", alpha=0.5, lw=0, zorder=1)

    loud_lo, loud_hi = np.nanpercentile(frames.loudness_db, [40, 99])
    # pitch is noisy frame to frame; a 250 ms running median keeps the melody, drops the jitter
    pitch = frames.pitch_hz.where(frames.voiced_prob > 0.1).rolling(25, center=True, min_periods=3).median()
    p_lo, p_hi = np.nanpercentile(pitch, [5, 95])

    t = frames.time.to_numpy()
    for w in words[(words.end > t0) & (words.start < t1)].itertuples():
        m = (t >= w.start) & (t <= w.end)
        if m.sum() < 2:
            continue
        x = t[m]
        thick = 0.04 + 0.55 * _norm(frames.loudness_db.to_numpy()[m], loud_lo, loud_hi)
        y = pitch.to_numpy()[m]
        fallback = w.pitch_mean if not np.isnan(w.pitch_mean) else (p_lo + p_hi) / 2
        y = np.where(np.isnan(y), fallback, y)
        y = pd.Series(y).rolling(9, center=True, min_periods=1).mean().to_numpy()
        y = (_norm(y, p_lo, p_hi) - 0.5) * 0.9
        # taper both ends like a brush lifting off
        taper = np.sin(np.linspace(0, np.pi, len(x))) ** 0.4
        th = thick * taper
        ax.fill_between(x, y - th / 2, y + th / 2, color=INK, alpha=0.88, lw=0, zorder=3)
        # dry-brush edge
        ax.fill_between(x, y - th / 2 - 0.02, y + th / 2 + 0.02, color=INK, alpha=0.12, lw=0, zorder=2)

    # seals: predicted applause invitations (last word of the sentence)
    if "invites_applause" in words:
        last = words[words.invites_applause == True].groupby("seg_id").tail(1)  # noqa: E712
        for w in last[(last.end > t0) & (last.end < t1)].itertuples():
            ax.add_patch(Rectangle((w.end - 0.25, -1.25), 0.5, 0.14, color=SEAL, alpha=0.85, lw=0, zorder=4))

    # sparse captions: first words of each sentence, very small
    last_caption = -np.inf
    for _, s in words.groupby("seg_id"):
        first = s.iloc[0]
        if t0 <= first.start < t1 and first.start - last_caption > (t1 - t0) / 25:
            last_caption = first.start
            txt = " ".join(s.word.head(4))
            ax.text(first.start, 1.18, txt, fontsize=5.5, color="#6b6355", family="serif",
                    va="center", ha="left", zorder=4)


def render(frames, words, events, path_long, path_preview, rows: int = 4) -> None:
    t0, t1 = words.start.min() - 1, events.end.max() if len(events) else words.end.max() + 1
    # long scroll: ~20 px per second of speech, short and wide
    fig = plt.figure(figsize=((t1 - t0) * 0.6, 3.2), dpi=120, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_scroll(ax, frames, words, events, t0, t1)
    fig.savefig(path_long, facecolor=PAPER)
    plt.close(fig)

    # preview: the same scroll cut into rows, like a hand scroll laid out on a table
    edges = np.linspace(t0, t1, rows + 1)
    fig, axes = plt.subplots(rows, 1, figsize=(16, 2.4 * rows), dpi=110, facecolor=PAPER)
    for ax, a, b in zip(axes, edges[:-1], edges[1:]):
        draw_scroll(ax, frames, words, events, a, b)
        ax.text(a, -1.3, f"{a:.0f} s", fontsize=7, color="#8a816f", family="serif", va="bottom")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, hspace=0.08)
    fig.savefig(path_preview, facecolor=PAPER)
    plt.close(fig)
