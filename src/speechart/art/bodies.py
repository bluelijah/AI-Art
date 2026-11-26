"""Speech Bodies: each speech as a solid made of stacked discs (the "anatomy of a speech").

From the group's Medium page: a sound-wave "body" to compare speeches by.
Every mark is data:

- time            runs left to right
- disc radius     loudness at that moment (relative to the speaker's own level),
                  so the body swells and narrows with the voice
- spine           the body's centre bends with the melodic gesture of each
                  breath group (pitch, smoothed to the line's shape)
- pauses          no discs: the body pinches to a thin thread
- disc colour     the emotion the AI hears in the sentence (consensus of 5 runs)
- gold body       the room: applause probability drawn as a second, ghostly
                  body behind the speaker's
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import EllipseCollection
from matplotlib.colors import to_rgb

from speechart.art.wiring import _contours, _gesture, load

PAPER = "#ece6ee"
INK = "#2b2530"
INK_2 = "#7d7484"
GOLD = "#d9a72e"
EMOTION = {  # soft, iridescent tones for a pale ground
    "ANGER": "#e0705f", "DEFIANCE": "#e8a061", "RESOLVE": "#d8c463", "FEAR": "#6fa8c9",
    "GRIEF": "#8f88d6", "DISBELIEF": "#c48fcf", "IRONY": "#c9ab8a", "HOPE": "#79c99a",
    "NEUTRAL": "#b3adb6",
}


def _body(d: dict, step_s: float = 0.06):
    """Discs along the speech: (t, radius 0..1, spine offset in semitones, rgb) per step."""
    f = _contours(d, hop=1)
    w = d["words"]
    t0 = w.start.min()
    app = d["events"][d["events"].label == "applause"]
    t1 = max(w.end.max(), app.end.max() if len(app) else 0)
    # about 4500 discs per speech whatever its length, so long speeches don't turn solid
    step_s = max(step_s, (t1 - t0) / 4500)
    grid = np.arange(t0, t1, step_s)

    loud = np.interp(grid, f.time, f.loud_rel.fillna(-40))
    # scale to this speaker's own range while speaking, so quiet and loud moments differ
    speaking = (grid >= w.start.min()) & (grid <= w.end.max())
    lo, hi = np.percentile(loud[speaking], [15, 99])
    radius = np.clip((loud - lo) / (hi - lo), 0, 1) ** 1.6

    spine = np.zeros_like(grid)
    in_speech = np.zeros(len(grid), bool)
    for g in d["groups"].itertuples():
        m = (grid >= g.start) & (grid <= g.end)
        if m.sum() < 2:
            continue
        fm = (f.time >= g.start) & (f.time <= g.end)
        gest = _gesture(f.time[fm].to_numpy(), f.pitch_st[fm].to_numpy())
        spine[m] = np.interp(grid[m], f.time[fm], gest) if fm.sum() >= 2 else 0
        in_speech[m] = True
    radius = np.where(in_speech, np.maximum(radius, 0.04), 0.0)
    spine = pd.Series(np.where(in_speech, spine, np.nan)).interpolate(limit_direction="both").fillna(0)
    spine = spine.rolling(5, center=True, min_periods=1).mean().to_numpy()

    seg = np.interp(grid, f.time, f.seg_id).round().astype(int)
    emo = dict(zip(d["sentences"].seg_id, d["sentences"].emotion.fillna("NEUTRAL")))
    rgb = np.array([to_rgb(EMOTION.get(emo.get(s, "NEUTRAL"), EMOTION["NEUTRAL"])) for s in seg])

    scores_path = d.get("scores_path")
    room = np.zeros_like(grid)
    if scores_path is not None and Path(scores_path).exists():
        sc = pd.read_csv(scores_path)
        room = np.interp(grid, sc.time, sc.applause)
    return grid, radius, spine, rgb, room


def draw_body(ax, d: dict, y0: float = 0.0, length: float = 10.0, height: float = 1.0, aspect: float = 0.07):
    """Draw one speech body along x in [0, length], centerd on y0, max radius `height`."""
    d = {**d, "scores_path": Path(__file__).resolve().parents[3] / "output" / d["sid"] / "reaction_scores.csv"}
    grid, radius, spine, rgb, room = _body(d)
    x = (grid - grid[0]) / (grid[-1] - grid[0]) * length
    y = y0 + spine * height * 0.05  # ±9 st bends the spine by ±0.45 of the max radius

    # the room: a ghostly gold body behind the speaker
    rr = room * height * 1.1
    keep = rr > 0.02
    if keep.any():
        ax.add_collection(EllipseCollection(widths=(rr[keep] * aspect * 2), heights=rr[keep] * 2, angles=0,
                                            units="xy", offsets=np.column_stack([x[keep], np.full(keep.sum(), y0)]),
                                            transOffset=ax.transData, facecolors=(*to_rgb(GOLD), 0.012),
                                            edgecolors=(*to_rgb(GOLD), 0.16), linewidths=0.35, zorder=1))

    # the voice: discs, drawn back to front so the body reads as a solid
    r = radius * height
    keep = r > 0
    # rims more than fills: stacked rings read as a ribbed, layered body
    face = np.c_[rgb[keep] * 0.9 + 0.1, np.full(keep.sum(), 0.10)]
    edge = np.c_[rgb[keep] * 0.75, np.full(keep.sum(), 0.75)]
    ax.add_collection(EllipseCollection(widths=r[keep] * aspect * 2, heights=r[keep] * 2, angles=0, units="xy",
                                        offsets=np.column_stack([x[keep], y[keep]]), transOffset=ax.transData,
                                        facecolors=face, edgecolors=edge, linewidths=0.45, zorder=2))
    # the thread that runs through pauses
    ax.plot(x, y, color=INK, lw=0.35, alpha=0.35, zorder=3)
    return x, y


def render_body(sid: str, path: Path) -> None:
    d = load(sid)
    fig = plt.figure(figsize=(18, 5.2), dpi=200, facecolor=PAPER)
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.78])
    ax.set_facecolor(PAPER)
    ax.axis("off")
    draw_body(ax, d, length=10, height=1.0)
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    m = d["meta"]
    fig.text(0.02, 0.93, m["speaker"], color=INK, fontsize=14, family="serif")
    fig.text(0.02, 0.895, f"{m['title']}, {m['date'][:4]}" + (" (excerpt)" if m.get("excerpt") else ""),
             color=INK_2, fontsize=9, family="serif")
    fig.text(0.98, 0.03, "time left to right   width: loudness   spine: melody of each breath   "
             "color: emotion the AI hears   gold: the room applauding", color=INK_2, fontsize=7.5,
             family="serif", ha="right")
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def render_bodies(ids: list[str], path: Path, title: str = "Speech Bodies") -> None:
    """All speeches stacked, each stretched to the same length so their shapes compare."""
    ds = sorted([load(s) for s in ids], key=lambda d: d["meta"]["date"])
    n = len(ds)
    fig = plt.figure(figsize=(18, 3.6 * n + 1.2), dpi=200, facecolor=PAPER)
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.88])
    ax.set_facecolor(PAPER)
    ax.axis("off")
    gap = 3.2
    for i, d in enumerate(ds):
        y0 = -i * gap
        draw_body(ax, d, y0=y0, length=10, height=1.0)
        m = d["meta"]
        minutes = (d["words"].end.max() - d["words"].start.min()) / 60
        ax.text(-0.3, y0 + 1.45, f"{m['speaker']}  ·  {m['title']}, {m['date'][:4]}"
                + (" (excerpt)" if m.get("excerpt") else "") + f"  ·  {minutes:.1f} min",
                color=INK_2, fontsize=9, family="serif", va="bottom")
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-(n - 1) * gap - 1.5, 1.95)
    ax.set_aspect("equal")
    fig.text(0.02, 0.965, title, color=INK, fontsize=18, family="serif")
    fig.text(0.98, 0.012, "Each speech stretched to the same length. Width: loudness. Spine: the melody of each "
             "breath. Color: the emotion the AI hears. Gold: the room applauding.", color=INK_2, fontsize=8,
             family="serif", ha="right")
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)
