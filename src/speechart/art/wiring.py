"""The Hidden Wiring of Speech: each speech drawn as its own structure.

Every mark comes from a structure found in the data (see docs/structures.md):

- form      : the speech spirals out from the centre, start to end (turns grow
              with the square root of its length)
- lines     : each breath group (the words between two pauses) is one stroke
              along the spiral; the speaker's real pitch contour bends it in
              and out, so the stroke *is* the melody of that line
- weight    : stroke width follows loudness, relative to the speaker's own level
- silence   : pauses are gaps in the spiral
- colour    : the emotion the AI hears in each sentence (consensus of 5 runs)
- wiring    : threads through the interior join sentences the AI finds
              echoing each other in meaning, far apart in time
- response  : gold halos where the room applauded
- sections  : (annotated version only) where the speech changes character

render_constellation() places several portraits together and draws the
echoes between speeches as threads crossing from one to another.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parents[3]
BG = "#07070b"
INK = "#e9e4d8"
INK_2 = "#8f8a80"
GOLD = "#f2c14e"
EMOTION = {  # luminous on a dark ground
    "ANGER": "#ff5b3d", "DEFIANCE": "#ff9d3a", "RESOLVE": "#e9d36b", "FEAR": "#5db7e3",
    "GRIEF": "#8c7ff0", "DISBELIEF": "#d08ee0", "IRONY": "#d9b48a", "HOPE": "#78e39a",
    "NEUTRAL": "#a9a59a",
}


# ---------------------------------------------------------------- geometry


class Spiral:
    """Archimedean spiral mapping speech time to a point, outward from the centre."""

    def __init__(self, t0: float, t1: float, turns: float, r0: float = 0.22, r1: float = 1.0):
        self.t0, self.t1, self.turns, self.r0, self.r1 = t0, t1, turns, r0, r1

    def u(self, t):
        return np.clip((np.asarray(t, float) - self.t0) / (self.t1 - self.t0), 0, 1)

    def point(self, t, offset=0.0):
        """(x, y) at time t, pushed `offset` along the outward normal."""
        u = self.u(t)
        theta = np.pi / 2 - 2 * np.pi * self.turns * u  # start at 12 o'clock, clockwise
        r = self.r0 + (self.r1 - self.r0) * u
        # outward normal of an Archimedean spiral is close to radial for many turns; use radial
        r = r + offset
        return np.cos(theta) * r, np.sin(theta) * r

    @property
    def ring_gap(self) -> float:
        return (self.r1 - self.r0) / self.turns


# ---------------------------------------------------------------- data


def load(sid: str) -> dict:
    out = ROOT / "output" / sid
    meta = json.loads((ROOT / "data" / "speeches" / sid / "meta.json").read_text())
    d = {"sid": sid, "meta": meta,
         "frames": pd.read_csv(out / "frames.csv"),
         "words": pd.read_csv(out / "words_timeline.csv"),
         "sentences": pd.read_csv(out / "sentences.csv"),
         "groups": pd.read_csv(out / "structure" / "breath_groups.csv"),
         "sections": pd.read_csv(out / "structure" / "sections.csv"),
         "events": pd.read_csv(out / "reaction_events.csv"),
         "levels": json.loads((out / "speaker_levels.json").read_text())}
    edges = ROOT / "output" / "compare" / "structure" / "echo_edges.csv"
    d["edges"] = pd.read_csv(edges) if edges.exists() else pd.DataFrame()
    return d


def _contours(d: dict, hop: int = 3):
    """Per-frame (time, pitch in semitones vs typical, loudness vs typical, seg_id) within breath groups."""
    f = d["frames"].iloc[::hop].reset_index(drop=True)
    lv = d["levels"]
    # a slow melody line: a 250 ms median removes octave jumps, then a 200 ms mean smooths
    pitch = (f.pitch_hz.where(f.voiced_prob > 0.1).rolling(9, center=True, min_periods=2).median())
    st = (12 * np.log2(pitch / lv["pitch_hz"])).clip(-9, 9)
    st = st.interpolate(limit=20, limit_direction="both")
    f["pitch_st"] = st.rolling(7, center=True, min_periods=1).mean()
    f["loud_rel"] = f.loudness_db.rolling(5, center=True, min_periods=1).mean() - lv["loudness_db"]
    w = d["words"]
    f["seg_id"] = -1
    for r in w.itertuples():
        f.loc[(f.time >= r.start) & (f.time <= r.end), "seg_id"] = r.seg_id
    # frames between two words of the same breath belong to the sentence being spoken
    f["seg_id"] = f.seg_id.replace(-1, np.nan).ffill().bfill().astype(int)
    return f


# ---------------------------------------------------------------- drawing


def _gesture(t: np.ndarray, pitch_st: np.ndarray, max_degree: int = 3) -> np.ndarray:
    """The melodic gesture of one breath group: a smooth low-order curve through its pitch.

    Syllable-level wiggles are dropped; what remains is the line's shape (rise,
    fall, arch), the same thing the gesture vocabulary clusters. Longer lines
    may bend more (degree grows with length, up to cubic).
    """
    ok = np.isfinite(pitch_st)
    if ok.sum() < 4:
        return np.zeros_like(t)
    x = (t - t[0]) / max(t[-1] - t[0], 1e-6)
    deg = int(min(max_degree, 1 + (t[-1] - t[0]) // 1.0, ok.sum() - 1))
    coef = np.polyfit(x[ok], pitch_st[ok], deg)
    return np.clip(np.polyval(coef, x), -9, 9)


def _sentence_mid(d: dict) -> dict:
    s = d["sentences"]
    return dict(zip(s.seg_id, (s.start + s.end) / 2))


def draw_portrait(ax, d: dict, cx: float = 0.0, cy: float = 0.0, scale: float = 1.0,
                  annotated: bool = False, glow: bool = True, thread_alpha: float = 1.0) -> Spiral:
    w = d["words"]
    t0 = w.start.min()
    t1 = max(w.end.max(), d["events"][d["events"].label == "applause"].end.max()
             if (d["events"].label == "applause").any() else w.end.max())
    minutes = (t1 - t0) / 60
    sp = Spiral(t0, t1, turns=1.25 * np.sqrt(minutes))
    gap = sp.ring_gap
    f = _contours(d)
    emo = dict(zip(d["sentences"].seg_id, d["sentences"].emotion.fillna("NEUTRAL")))
    P = lambda x, y: (cx + x * scale, cy + y * scale)

    # wiring first, so the voice sits on top
    mid = _sentence_mid(d)
    e = d["edges"]
    if len(e):
        inside = e[(e.kind == "within") & (e.speech_a == d["sid"]) & (e.time_gap_s > 20)]
        density = min(1.0, 40 / max(len(inside), 1)) ** 0.5 * thread_alpha
        for r in inside.itertuples():
            if r.seg_a not in mid or r.seg_b not in mid:
                continue
            xa, ya = sp.point(mid[r.seg_a])
            xb, yb = sp.point(mid[r.seg_b])
            # quadratic curve bowed toward the centre: the deeper the bow, the further apart in time
            k = 0.25 + 0.5 * min(1.0, abs(mid[r.seg_b] - mid[r.seg_a]) / (t1 - t0))
            mx, my = (xa + xb) / 2 * (1 - k), (ya + yb) / 2 * (1 - k)
            tt = np.linspace(0, 1, 60)[:, None]
            pts = (1 - tt) ** 2 * np.array([xa, ya]) + 2 * (1 - tt) * tt * np.array([mx, my]) + tt ** 2 * np.array([xb, yb])
            ca = np.array(to_rgb(EMOTION.get(emo.get(r.seg_a, "NEUTRAL"), EMOTION["NEUTRAL"])))
            cb = np.array(to_rgb(EMOTION.get(emo.get(r.seg_b, "NEUTRAL"), EMOTION["NEUTRAL"])))
            cols = (1 - tt) * ca + tt * cb
            strength = np.clip((r.similarity - 0.45) / 0.45, 0.1, 1)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            segs = np.stack([np.stack(P(*s.T), axis=-1) for s in segs])
            ax.add_collection(LineCollection(segs, colors=np.c_[cols[:-1], np.full(len(segs), (0.06 + 0.10 * strength) * density)],
                                             linewidths=(2.5 + 5 * strength) * scale, capstyle="round", zorder=1))
            ax.add_collection(LineCollection(segs, colors=np.c_[cols[:-1], np.full(len(segs), (0.30 + 0.45 * strength) * density)],
                                             linewidths=(0.5 + 1.1 * strength) * scale, capstyle="round", zorder=1))

    # applause halos
    for ev in d["events"][(d["events"].label == "applause") & (d["events"].end > t0)].itertuples():
        tt = np.linspace(max(ev.start, t0), min(ev.end, t1), 40)
        x, y = sp.point(tt)
        X, Y = P(x, y)
        for k, a in [(26, 0.04), (16, 0.07), (8, 0.12)]:
            ax.plot(X, Y, color=GOLD, lw=k * scale * (0.6 + 0.4 * ev.peak), alpha=a, solid_capstyle="round", zorder=2)
        ax.plot(X, Y, color=GOLD, lw=1.6 * scale, alpha=0.9, solid_capstyle="round", zorder=5)

    # the voice: one stroke per breath group, bent by its pitch contour
    amp = gap * 0.04  # displacement per semitone (±9 st stays inside the ring gap)
    for g in d["groups"].itertuples():
        m = (f.time >= g.start) & (f.time <= g.end)
        if m.sum() < 3:
            continue
        seg = f[m]
        pitch = _gesture(seg.time.to_numpy(), seg.pitch_st.to_numpy())
        x, y = sp.point(seg.time.to_numpy(), offset=pitch * amp)
        X, Y = P(x, y)
        loud = np.clip((seg.loud_rel.to_numpy() + 18) / 24, 0.02, 1)
        taper = np.sin(np.linspace(0, np.pi, len(X))) ** 0.35
        width = (0.25 + 3.2 * loud * taper) * scale
        cols = np.array([to_rgb(EMOTION.get(emo.get(s, "NEUTRAL"), EMOTION["NEUTRAL"])) for s in seg.seg_id])
        pts = np.column_stack([X, Y])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        if glow:
            ax.add_collection(LineCollection(segs, colors=np.c_[cols[:-1], np.full(len(segs), 0.10)],
                                             linewidths=width[:-1] * 4, capstyle="round", zorder=3))
        ax.add_collection(LineCollection(segs, colors=np.c_[cols[:-1], np.full(len(segs), 0.95)],
                                         linewidths=width[:-1], capstyle="round", zorder=4))

    if annotated:
        for s in d["sections"].iloc[1:].itertuples():
            if not isinstance(s.opens_with, str):
                continue
            (xa, ya), (xb, yb) = sp.point(s.start, -gap * 0.45), sp.point(s.start, gap * 0.45)
            ax.plot(*P(np.array([xa, xb]), np.array([ya, yb])), color=INK_2, lw=0.6 * scale, alpha=0.8, zorder=6)
            tx, ty = sp.point(s.start, gap * 0.55)
            words = " ".join(s.opens_with.split()[:5])
            ax.text(*P(tx, ty), words + ("…" if len(s.opens_with.split()) > 5 else ""), color=INK_2,
                    fontsize=5.5 * scale, family="serif", ha="center", va="center", zorder=6)
    return sp


def _frame(size: float = 12):
    fig = plt.figure(figsize=(size, size), dpi=200, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _legend(fig, emotions_present, y=0.03):
    keys = [k for k in EMOTION if k in emotions_present]
    for i, k in enumerate(keys):
        x = 0.04 + i * 0.095
        fig.patches.append(plt.Circle((x, y + 0.006), 0.004, color=EMOTION[k], transform=fig.transFigure, figure=fig))
        fig.text(x + 0.008, y, k.lower(), color=INK_2, fontsize=7, family="serif")


def render_portrait(sid: str, path: Path, annotated: bool = False) -> None:
    d = load(sid)
    fig, ax = _frame()
    draw_portrait(ax, d, annotated=annotated)
    lim = 1.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    m = d["meta"]
    fig.text(0.04, 0.965, f"{m['speaker']}", color=INK, fontsize=15, family="serif")
    fig.text(0.04, 0.945, f"{m['title']}, {m['date'][:4]}" + (" (excerpt)" if m.get("excerpt") else ""),
             color=INK_2, fontsize=9, family="serif")
    if annotated:
        fig.text(0.96, 0.965, "spiral: the speech from centre outward   stroke: one breath, bent by its pitch,\n"
                 "width = loudness   colour: emotion the AI hears   threads: sentences echoing in meaning\n"
                 "gold: applause   marks: where the speech changes character",
                 color=INK_2, fontsize=6.5, family="serif", ha="right", va="top")
        _legend(fig, set(d["sentences"].emotion.dropna()))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def render_constellation(ids: list[str], path: Path, title: str = "The Hidden Wiring of Speech") -> None:
    """All portraits on one sheet; echoes between speeches drawn as threads crossing between them.

    Speeches sit on a ring in chronological order. A cross-speech thread joins
    the two sentences the AI places closest in meaning, so speakers who never
    cite each other are visibly wired together.
    """
    ds = sorted([load(s) for s in ids], key=lambda d: d["meta"]["date"])
    n = len(ds)
    R, scale = 1.55, 0.78
    fig, ax = _frame(16)
    centres, spirals = {}, {}
    for i, d in enumerate(ds):
        a = np.pi / 2 + np.pi / n - 2 * np.pi * i / n
        cx, cy = R * np.cos(a), R * np.sin(a)
        spirals[d["sid"]] = draw_portrait(ax, d, cx, cy, scale, thread_alpha=0.55)
        centres[d["sid"]] = (cx, cy, d)
        m = d["meta"]
        ty = cy + (1.13 if cy > 0 else -1.13) * scale
        ax.text(cx, ty, f"{m['speaker']}\n{m['title']}, {m['date'][:4]}" + (" (excerpt)" if m.get("excerpt") else ""),
                color=INK_2, fontsize=8, family="serif", ha="center", va="bottom" if cy > 0 else "top")

    e = ds[0]["edges"]
    across = e[e.kind == "across"] if len(e) else e
    mids = {d["sid"]: _sentence_mid(d) for d in ds}
    for r in across.itertuples():
        if r.speech_a not in centres or r.speech_b not in centres:
            continue
        pts = []
        for sid, seg in [(r.speech_a, r.seg_a), (r.speech_b, r.seg_b)]:
            cx, cy, d = centres[sid]
            x, y = spirals[sid].point(mids[sid][seg])
            pts.append(np.array([cx + x * scale, cy + y * scale]))
        a, b = pts
        ctrl = (a + b) / 2 * 0.25  # bow every thread through the middle of the sheet
        tt = np.linspace(0, 1, 120)[:, None]
        curve = (1 - tt) ** 2 * a + 2 * (1 - tt) * tt * ctrl + tt ** 2 * b
        strength = np.clip((r.similarity - 0.45) / 0.45, 0.1, 1)
        ax.plot(*curve.T, color=INK, lw=6 * strength, alpha=0.05 + 0.06 * strength, solid_capstyle="round", zorder=7)
        ax.plot(*curve.T, color=INK, lw=0.5 + 1.3 * strength, alpha=0.35 + 0.5 * strength, solid_capstyle="round",
                zorder=7)
        for p in (a, b):
            ax.scatter(*p, s=18 * strength + 6, color=INK, alpha=0.9, zorder=8, linewidths=0)

    lim = R + 1.12 * scale
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim - 0.08, lim + 0.04)
    fig.text(0.03, 0.975, title, color=INK, fontsize=18, family="serif", va="top")
    fig.text(0.03, 0.055, "Each spiral is one speech, centre outward. Each stroke is one breath, bent by its pitch; "
             "width is loudness; colour is the emotion the AI hears.\nColoured threads: sentences echoing each other "
             "inside a speech. White threads: echoes between speeches, found by an AI that places every sentence in a "
             "space of meaning. Gold: applause.", color=INK_2, fontsize=8.5, family="serif", va="bottom")
    _legend(fig, set().union(*[set(d["sentences"].emotion.dropna()) for d in ds]), y=0.02)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
