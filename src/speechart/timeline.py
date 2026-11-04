"""The shared timeline: every signal on one time axis.

Two views of the same speech:

- frame grid  : one row per time step (10 ms from delivery.py, or coarser
                via `resample`). Continuous signals live here, and interval
                events (pauses, applause, device spans) become 0/1 columns.
                This is what lead/lag analysis runs on.
- word table  : one row per spoken word, with the delivery features of that
                word attached. This is what most artworks draw from
                (e.g. Breath Scroll: stroke weight = loudness, gap = pause_after).

Word input schema (what WhisperX produces): columns `word`, `start`, `end`
in seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mark_intervals(frames: pd.DataFrame, intervals: pd.DataFrame, name: str) -> pd.DataFrame:
    """Add a 0/1 column `name` to the frame grid: 1 where any interval covers the frame.

    `intervals` needs `start` and `end` columns (seconds).
    """
    t = frames["time"].to_numpy()
    mask = np.zeros(len(t), dtype=np.int8)
    for start, end in intervals[["start", "end"]].itertuples(index=False):
        mask[(t >= start) & (t < end)] = 1
    out = frames.copy()
    out[name] = mask
    return out


def resample(frames: pd.DataFrame, step_s: float) -> pd.DataFrame:
    """Coarsen the grid to `step_s` bins (mean of each column per bin).

    Pitch is averaged over voiced frames only; 0/1 columns become the
    fraction of the bin they cover.
    """
    bins = np.floor(frames["time"] / step_s).astype(int)
    out = frames.drop(columns="time").groupby(bins).mean(numeric_only=True)
    out.insert(0, "time", out.index * step_s)
    return out.reset_index(drop=True)


def words_with_delivery(words: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Attach delivery features to each word.

    Adds columns:
      loudness_mean, loudness_max : dB over the word's span
      pitch_mean                  : Hz over voiced frames in the span (NaN if none)
      pause_after                 : silence before the next word starts (s);
                                    NaN for the last word
    """
    t = frames["time"].to_numpy()
    loud = frames["loudness_db"].to_numpy()
    pitch = frames["pitch_hz"].to_numpy()

    w = words.sort_values("start").reset_index(drop=True).copy()
    lo = np.searchsorted(t, w["start"].to_numpy(), side="left")
    hi = np.searchsorted(t, w["end"].to_numpy(), side="right")

    loud_mean, loud_max, pitch_mean = [], [], []
    for a, b in zip(lo, hi):
        b = max(b, a + 1)  # very short words still get one frame
        seg_l = loud[a:b]
        seg_p = pitch[a:b]
        seg_p = seg_p[~np.isnan(seg_p)]
        loud_mean.append(float(seg_l.mean()) if len(seg_l) else np.nan)
        loud_max.append(float(seg_l.max()) if len(seg_l) else np.nan)
        pitch_mean.append(float(seg_p.mean()) if len(seg_p) else np.nan)

    w["loudness_mean"] = loud_mean
    w["loudness_max"] = loud_max
    w["pitch_mean"] = pitch_mean
    w["pause_after"] = (w["start"].shift(-1) - w["end"]).clip(lower=0)
    return w


def lagged_correlation(a: pd.Series, b: pd.Series, max_lag: int) -> pd.Series:
    """Correlation of a(t) with b(t + lag) for lag in [-max_lag, max_lag] steps.

    A peak at a positive lag means `a` tends to come *before* `b`
    (e.g. a = pause after a contrast phrase, b = applause; peak at +20
    on a 100 ms grid = applause about 2 s later).
    """
    a = pd.Series(np.asarray(a, dtype=float))
    b = pd.Series(np.asarray(b, dtype=float))
    lags = range(-max_lag, max_lag + 1)
    return pd.Series([a.corr(b.shift(-lag)) for lag in lags], index=list(lags), name="r")


def clip_long_words(words: pd.DataFrame, max_word_s: float = 1.2) -> pd.DataFrame:
    """Fix words the aligner stretched over applause or silence.

    Forced alignment sometimes glues a word to the gap before it (e.g. "The"
    spanning 7 s of applause). The word's end is usually right, so keep the
    end and give it a plausible length based on its characters.
    """
    w = words.copy()
    dur = w["end"] - w["start"]
    est = 0.07 * w["word"].str.len() + 0.10
    bad = dur > max_word_s
    w.loc[bad, "start"] = w.loc[bad, "end"] - np.minimum(dur[bad], est[bad])
    w["clipped"] = bad
    return w


def reaction_after(words: pd.DataFrame, events: pd.DataFrame, window_s: float = 3.0) -> pd.DataFrame:
    """For each word, the reaction (if any) that starts within `window_s` after it ends.

    Adds: reaction_after (label or ""), reaction_delay (s), reaction_duration (s).
    A reaction is credited only to the last word before it starts.
    """
    w = words.copy()
    w["reaction_after"] = ""
    w["reaction_delay"] = np.nan
    w["reaction_duration"] = np.nan
    ends = w["end"].to_numpy()
    for ev in events.itertuples(index=False):
        before = np.flatnonzero(ends <= ev.start + 0.25)  # small tolerance for window edges
        if len(before) == 0:
            continue
        i = before[-1]
        delay = ev.start - ends[i]
        if delay <= window_s:
            w.loc[i, ["reaction_after", "reaction_delay", "reaction_duration"]] = [ev.label, max(delay, 0.0), ev.duration]
    return w
