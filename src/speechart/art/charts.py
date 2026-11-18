"""Analysis charts: the plain, readable views behind the artworks.

- how_dare_you : every repetition of a phrase, overlaid and aligned on its
                 first word, so delivery can be compared like for like
- anatomy      : sentence endings aligned on the last word, split by whether
                 applause followed, so we can see what precedes applause

Colours follow the dataviz reference palette (fixed categorical order,
text in ink colours, one y-scale per panel, never a dual axis).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _style(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def smoothed_frames(frames: pd.DataFrame) -> pd.DataFrame:
    f = frames.copy()
    f["loud_s"] = f.loudness_db.rolling(9, center=True, min_periods=1).mean()
    f["pitch_s"] = (f.pitch_hz.where(f.voiced_prob > 0.1)
                    .rolling(25, center=True, min_periods=3).median())
    return f


def find_phrase(words: pd.DataFrame, phrase: str) -> list[int]:
    """Row indices where `phrase` (case/punctuation-insensitive) starts."""
    toks = phrase.lower().split()
    wl = words.word.str.lower().str.strip("?,.!;:\"'").tolist()
    return [i for i in range(len(wl) - len(toks) + 1) if wl[i:i + len(toks)] == toks]


def how_dare_you(frames, words, scores, events, path, phrase="how dare you", before_s=1.0, after_s=4.5):
    f = smoothed_frames(frames)
    starts = find_phrase(words, phrase)
    n_tok = len(phrase.split())

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True, dpi=150, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 3, 1.6], "hspace": 0.18})
    labels, ends_rel = [], []
    for k, i in enumerate(starts):
        t0 = words.start[i]
        end_rel = words.end[i + n_tok - 1] - t0
        phrase_end = words.end[i + n_tok - 1]
        app = events[events.label == "applause"]
        got = ((app.start >= phrase_end - 0.25) & (app.start <= phrase_end + 3)).any()
        c = SERIES[k]
        lw = 2.6 if got else 1.6
        seg = f[(f.time >= t0 - before_s) & (f.time <= t0 + after_s)]
        x = seg.time - t0
        axes[0].plot(x, seg.loud_s, color=c, lw=lw)
        axes[1].plot(x, seg.pitch_s, color=c, lw=lw)
        sc = scores[(scores.time >= t0 - before_s) & (scores.time <= t0 + after_s)]
        axes[2].plot(sc.time - t0, sc.applause, color=c, lw=lw)
        labels.append(f"#{k + 1} at {t0 // 60:.0f}:{t0 % 60:02.0f}" + ("  → applause" if got else ""))
        ends_rel.append(end_rel)

    _style(axes[0], "loudness (dB)")
    _style(axes[1], "pitch (Hz)")
    _style(axes[2], "applause\nprobability")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel(f"seconds from the start of “{phrase.capitalize()}”", color=INK_2, fontsize=9)
    axes[0].set_xlim(-before_s, after_s)
    handles = [plt.Line2D([], [], color=SERIES[k], lw=2) for k in range(len(starts))]
    for ax in axes:
        ax.axvspan(0, float(np.mean(ends_rel)), color=GRID, alpha=0.5, lw=0, zorder=0)
    axes[0].text(float(np.mean(ends_rel)) / 2, 1.01, "the phrase", transform=axes[0].get_xaxis_transform(),
                 ha="center", fontsize=8, color=INK_2)
    axes[0].legend(handles, labels, ncol=2, frameon=False,
                   loc="upper left", bbox_to_anchor=(0, 1.2), fontsize=8, labelcolor=INK_2)
    fig.suptitle(f"The {len(starts)} times she says “{phrase.capitalize()}”", x=0.06, ha="left",
                 color=INK, fontsize=13, y=0.97)
    fig.subplots_adjust(left=0.1, right=0.96, top=0.86, bottom=0.07)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def anatomy(frames, words, path, before_s=8.0, after_s=4.0):
    """Every sentence ending, aligned on its last word; applause vs. no applause."""
    f = smoothed_frames(frames)
    t = f.time.to_numpy()
    hop = t[1] - t[0]
    rel = np.round(np.arange(-before_s, after_s + 1e-9, hop), 3)
    loud, pitch = f.loud_s.to_numpy(), f.pitch_s.to_numpy()

    def sample(arr, times):
        idx = np.round((times - t[0]) / hop).astype(int)
        ok = (idx >= 0) & (idx < len(arr))
        out = np.full(len(times), np.nan)
        out[ok] = arr[idx[ok]]
        return out

    ends = words.groupby("seg_id").tail(1)
    groups = {"followed by applause": [], "no applause": []}
    for w in ends.itertuples():
        key = "followed by applause" if w.reaction_after == "applause" else "no applause"
        grid_t = w.end + rel
        groups[key].append((sample(loud, grid_t), sample(pitch, grid_t)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, dpi=150, facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.15})
    colours = {"followed by applause": SERIES[1], "no applause": SERIES[0]}
    for key, rows in groups.items():
        if not rows:
            continue
        c = colours[key]
        for j, ax in enumerate(axes):
            arr = np.array([r[j] for r in rows])
            for r in arr:
                ax.plot(rel, r, color=c, lw=0.6, alpha=0.18 if key == "no applause" else 0.35)
            with np.errstate(all="ignore"):
                # average only where at least 2 sentences have a value (pitch has gaps)
                mean = np.where(np.sum(~np.isnan(arr), 0) >= 2, np.nanmean(arr, 0), np.nan)
            ax.plot(rel, mean, color=c, lw=2.4)
    handles = [plt.Line2D([], [], color=colours[k], lw=2.4) for k in groups]
    axes[0].legend(handles, [f"{k} (n={len(v)})" for k, v in groups.items()], frameon=False, ncol=2,
                   loc="upper left", bbox_to_anchor=(0, 1.14), fontsize=8, labelcolor=INK_2)
    for ax in axes:
        ax.axvline(0, color=INK_2, lw=0.8, ls=(0, (2, 2)))
    _style(axes[0], "loudness (dB)")
    _style(axes[1], "pitch (Hz)")
    axes[1].set_xlabel("seconds from the end of the sentence's last word", color=INK_2, fontsize=9)
    axes[0].set_xlim(-before_s, after_s)
    fig.suptitle("What comes before applause: every sentence ending, aligned", x=0.06, ha="left",
                 color=INK, fontsize=13, y=0.97)
    fig.text(0.06, 0.925, "thin lines: single sentences    thick lines: average", color=INK_2, fontsize=8)
    fig.subplots_adjust(left=0.1, right=0.96, top=0.84, bottom=0.09)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
