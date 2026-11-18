"""Applause Landscape: the speech as a painted mountain panorama.

Read left to right like a hand scroll. Data sets the geography; the only
randomness is fine ridge texture from a fixed seed:
- front range height = how the room responded (applause probability):
                       peaks stand exactly where the audience reacted
- back range height  = how loudly she spoke (smoothed loudness)
- ridge colour       = emotional tone of the sentence (blind LLM tags)
- water at the base  = pauses: the longer the silence, the wider the water

It also writes a 16-bit depth map (near = bright), so a diffusion model with
depth conditioning (e.g. ControlNet depth) can paint texture onto exactly
this geography later without moving anything.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from PIL import Image

EMOTION_COLOURS = {  # painterly, earthy (art piece, not a chart palette)
    "ANGER": "#b5452c", "DEFIANCE": "#c9792e", "RESOLVE": "#8f8a3c", "FEAR": "#4f6d7a",
    "GRIEF": "#5b5a7d", "DISBELIEF": "#8a6f8e", "IRONY": "#a58a5a", "HOPE": "#6f9a5b",
    "NEUTRAL": "#a99f86",
}
SKY_TOP, SKY_LOW = np.array(to_rgb("#efe6cf")), np.array(to_rgb("#f6f0e2"))
MIST = np.array(to_rgb("#e9e1cc"))
WATER = np.array(to_rgb("#6f8f99"))
PAPER = "#efe8d6"
GOLD = "#c9a227"


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    k = np.hanning(max(n, 3))
    k /= k.sum()
    pad = len(k)
    xp = np.pad(np.nan_to_num(x), pad, mode="edge")
    return np.convolve(xp, k, mode="same")[pad:-pad]


def _norm(x):
    lo, hi = np.nanpercentile(x, [3, 99.5])
    return np.clip((x - lo) / (hi - lo), 0, 1)


def emotion_strip(xs: np.ndarray, words: pd.DataFrame) -> np.ndarray:
    rgb = np.tile(np.array(to_rgb(EMOTION_COLOURS["NEUTRAL"])), (len(xs), 1))
    for _, s in words.groupby("seg_id"):
        emo = s.emotion.iloc[0] if "emotion" in s and isinstance(s.emotion.iloc[0], str) else "NEUTRAL"
        m = (xs >= s.start.min() - 0.3) & (xs <= s.end.max() + 0.3)
        rgb[m] = to_rgb(EMOTION_COLOURS.get(emo, EMOTION_COLOURS["NEUTRAL"]))
    for c in range(3):
        rgb[:, c] = _smooth(rgb[:, c], 21)
    return rgb


def build(grid: pd.DataFrame, words: pd.DataFrame, width: int, height: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    t = grid.time.to_numpy()
    xs = np.linspace(t[0], t[-1], width)
    loud = _norm(np.interp(xs, t, grid.loudness_db.fillna(grid.loudness_db.min())))
    applause = np.interp(xs, t, grid.applause_prob.fillna(0))
    pause = np.interp(xs, t, grid.pause.fillna(0))

    detail = _smooth(rng.standard_normal(width), 5) * 0.012 + _smooth(rng.standard_normal(width), 31) * 0.03
    back = 0.30 + 0.22 * _smooth(loud, 61) + detail * 0.6                    # fraction of height
    front = 0.12 + 0.62 * _smooth(applause, 45) + 0.10 * _smooth(loud, 25) + detail
    water = 0.02 + 0.06 * _smooth(pause, 61)

    # ridge colour: the emotion being spoken, turning to gold where the room responds
    gold_mix = np.clip(_smooth(applause, 45) * 1.6, 0, 1)[:, None]
    colour = emotion_strip(xs, words) * (1 - gold_mix) + np.array(to_rgb(GOLD)) * gold_mix
    y = np.linspace(1, 0, height)[:, None]  # 1 = top of image
    img = SKY_LOW + (SKY_TOP - SKY_LOW) * y[..., None] ** 0.6
    img = np.broadcast_to(img, (height, width, 3)).copy()
    depth = np.zeros((height, width))

    # back range: pale, hazy (atmospheric perspective)
    below_b = y < back[None, :]
    fade_b = np.clip((back[None, :] - y) / 0.25, 0, 1)[..., None]
    back_col = (colour * 0.35 + MIST * 0.65)[None, :, :]
    img = np.where(below_b[..., None], back_col * (1 - 0.25 * fade_b) + MIST * 0.25 * fade_b, img)
    depth[below_b] = 0.35

    # front range: full colour at the ridge, sinking into mist toward the base
    below_f = y < front[None, :]
    fade_f = np.clip((front[None, :] - y) / 0.35, 0, 1)[..., None]
    ridge_col = colour[None, :, :]
    shade = 1 - 0.18 * np.clip((front[None, :] - y) / 0.05, 0, 1)[..., None]  # darker just under the ridge line
    front_rgb = ridge_col * shade * (1 - 0.6 * fade_f) + MIST * 0.6 * fade_f
    img = np.where(below_f[..., None], front_rgb, img)
    depth[below_f] = 0.7 + 0.3 * (1 - y[np.newaxis].repeat(1, 0)[0] * np.ones((1, width)))[below_f]

    # water at the base where she paused
    in_water = y < water[None, :]
    ripple = 0.06 * np.sin(np.linspace(0, 900, width))[None, :] * np.sin(np.linspace(0, 60, height))[:, None]
    img = np.where(in_water[..., None], WATER * (0.92 + ripple[..., None]) + 0.08 * MIST, img)
    depth[in_water] = 1.0

    # ink line along the front ridge
    ridge_px = ((1 - front) * (height - 1)).astype(int)
    for dx in range(width):
        r = ridge_px[dx]
        img[max(r - 1, 0):r + 1, dx] *= 0.55
    return np.clip(img, 0, 1), depth, xs


def render(grid: pd.DataFrame, words: pd.DataFrame, events: pd.DataFrame, path: str, depth_path: str,
           width: int = 3000, height: int = 900, title: str | None = None) -> None:
    img, depth, xs = build(grid, words, width, height)

    fig = plt.figure(figsize=(width / 170, height / 170 + 1.0), dpi=170, facecolor=PAPER)
    ax = fig.add_axes([0.015, 0.13, 0.97, 0.78])
    ax.imshow(img, aspect="auto", interpolation="bilinear")
    ax.axis("off")
    for ev in events[events.label == "applause"].itertuples():
        x = (ev.start + ev.duration / 2 - xs[0]) / (xs[-1] - xs[0]) * width
        ax.text(x, height * 0.05, f"{ev.start // 60:.0f}:{ev.start % 60:02.0f}", ha="center",
                fontsize=7, family="serif", color="#5a4f3d")

    fig.text(0.015, 0.95, title or "Applause Landscape", fontsize=13, family="serif", color="#2d261c")
    present = set(words["emotion"].dropna()) if "emotion" in words else set()
    keys = [k for k in EMOTION_COLOURS if k in present]
    for k, name in enumerate(keys):
        x0 = 0.015 + k * 0.075
        fig.patches.append(plt.Rectangle((x0, 0.045), 0.01, 0.03, color=EMOTION_COLOURS[name],
                                         transform=fig.transFigure, figure=fig))
        fig.text(x0 + 0.014, 0.05, name.lower(), fontsize=8, family="serif", color="#2d261c")
    fig.text(0.985, 0.05, "peaks (gold): audience response   far hills: loudness   water: pauses   colour: emotion",
             ha="right", fontsize=8, family="serif", color="#6b604c")
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)

    Image.fromarray((np.clip(depth, 0, 1) * 65535).astype(np.uint16)).save(depth_path)
