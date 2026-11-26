"""Ripples: the speech at the centre, its echo spreading outward in time.

- centre ring  = the speech itself, read clockwise from the top: stroke
                 weight follows loudness, gold where the room applauded,
                 white ticks where the tracked phrase was spoken
- outer rings  = days after the speech (one ring per day, a week, a month
                 labelled). Each full ring's brightness is attention to the
                 speaker (Wikipedia pageviews that day, log scale).
- gold arcs    = news articles quoting the tracked phrase (GDELT) that day,
                 drawn only at the angles where the phrase was spoken, so each
                 echo is traced back to the moment that sent it

Framed as resonance, not impact: the rings show what followed, not what
the speech caused.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

BG = "#0d0f14"
RING = "#9cc3e6"
GOLD = "#e0b43a"
INK = "#e8e6e1"
INK_2 = "#8d8a84"


def _angle(t, t0, t1):
    """Clockwise from 12 o'clock, in radians for matplotlib's anticlockwise convention."""
    return np.pi / 2 - 2 * np.pi * (np.asarray(t) - t0) / (t1 - t0)


def render(frames: pd.DataFrame, words: pd.DataFrame, events: pd.DataFrame,
           wiki: pd.DataFrame, gdelt: pd.DataFrame | None, speech_day: str, path: str,
           article: str, phrase_starts: list[float], phrase: str, title: str | None = None,
           days: int = 60) -> None:
    t0, t1 = words.start.min() - 0.5, max(words.end.max(), events.end.max() if len(events) else 0) + 0.5
    fig = plt.figure(figsize=(10, 10.8), dpi=170, facecolor=BG)
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.86])
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")

    r0, dr = 1.0, 0.055
    R = r0 + dr * (days + 2)
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)

    # centre: the speech as a ring of ink, weight = loudness
    f = frames[(frames.time >= t0) & (frames.time <= t1)].iloc[::5]
    loud = f.loudness_db.rolling(5, center=True, min_periods=1).mean().to_numpy()
    w = np.clip((loud - np.percentile(loud, 30)) / (np.percentile(loud, 99) - np.percentile(loud, 30)), 0, 1)
    a = _angle(f.time.to_numpy(), t0, t1)
    pts = np.column_stack([np.cos(a), np.sin(a)]) * (r0 * 0.78)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    ax.add_collection(LineCollection(segs, linewidths=0.4 + 5 * w[:-1], colors=INK, alpha=0.9, capstyle="butt"))
    for ev in events[events.label == "applause"].itertuples():
        aa = _angle(np.linspace(max(ev.start, t0), min(ev.end, t1), 60), t0, t1)
        ax.plot(np.cos(aa) * r0 * 0.9, np.sin(aa) * r0 * 0.9, color=GOLD, lw=5, solid_capstyle="round")
    for ps in phrase_starts:
        aa = _angle(ps, t0, t1)
        ax.plot([np.cos(aa) * r0 * 0.66, np.cos(aa) * r0 * 0.98], [np.sin(aa) * r0 * 0.66, np.sin(aa) * r0 * 0.98],
                color="white", lw=1.2)

    # rings: days after the speech
    d0 = pd.Timestamp(speech_day)
    views = wiki[wiki.article == article].set_index("date").views
    views.index = pd.to_datetime(views.index)
    base = views[(views.index < d0) & (views.index >= d0 - pd.Timedelta(days=14))].median()
    lv = np.log10(views / base).clip(lower=0)
    lv_max = lv.max() or 1
    theta = np.linspace(0, 2 * np.pi, 720)
    for d in range(days + 1):
        day = d0 + pd.Timedelta(days=d)
        r = r0 + dr * (d + 1)
        v = lv.get(day, 0.0) / lv_max
        ax.plot(np.cos(theta) * r, np.sin(theta) * r, color=RING, lw=0.3 + 2.2 * v, alpha=0.08 + 0.85 * v)

    # gold arcs: the phrase in the news, anchored where it was spoken
    if gdelt is not None and len(gdelt):
        g = gdelt.copy()
        g["date"] = pd.to_datetime(g["date"])
        g = g.set_index("date").articles
        g_max = g.max() or 1
        for d in range(days + 1):
            n = g.get(d0 + pd.Timedelta(days=d), 0)
            if n <= 0:
                continue
            r = r0 + dr * (d + 1)
            v = np.sqrt(n / g_max)
            span = np.deg2rad(2 + 9 * v)
            for ps in phrase_starts:
                c = _angle(ps, t0, t1)
                aa = np.linspace(c - span, c + span, 40)
                ax.plot(np.cos(aa) * r, np.sin(aa) * r, color=GOLD, lw=0.6 + 3.2 * v, alpha=0.35 + 0.6 * v,
                        solid_capstyle="round")

    for d, label in [(1, "next day"), (7, "1 week"), (30, "1 month"), (days, f"{days} days")]:
        r = r0 + dr * (d + 1)
        ax.text(r * np.cos(np.deg2rad(-58)), r * np.sin(np.deg2rad(-58)), label, color=INK_2, fontsize=7,
                family="serif", rotation=32, ha="left", va="center")
    ax.text(0, 0, "the\nspeech", color=INK_2, fontsize=9, family="serif", ha="center", va="center")

    fig.text(0.04, 0.955, title or "Ripples", color=INK, fontsize=14, family="serif")
    fig.text(0.04, 0.03,
             f"center: the speech, clockwise from the top (weight = loudness, gold = applause, white = “{phrase}”)\n"
             f"blue rings: Wikipedia views of {article.replace('_', ' ')} per day   "
             f"gold arcs: news articles quoting “{phrase}”, anchored where it was said",
             color=INK_2, fontsize=7.5, family="serif")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
