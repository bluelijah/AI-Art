"""Process data: the per-item records behind every summary number and chart.

The charts and metrics.csv show results; these tables show how each result
was reached, one row per sentence ending or per audience reaction, so any
claim can be traced back to the moments it came from.

- ending_stats   : delivery in the run-up to every sentence ending
                   (the inputs to "anatomy of applause")
- runup_curves   : the aligned loudness / pitch curves themselves, long format
- reaction_moments: every detected reaction with the sentence before it,
                   its tags, the AI's prediction and the delivery run-up

Levels are relative to the speaker's own typical level (median over the
main speaker's span), so different voices and recordings compare fairly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOP_S = 0.05  # resolution of the stored run-up curves


def _smooth(frames: pd.DataFrame) -> pd.DataFrame:
    f = frames.copy()
    f["loud_s"] = f.loudness_db.rolling(9, center=True, min_periods=1).mean()
    f["pitch_s"] = (f.pitch_hz.where(f.voiced_prob > 0.1)
                    .rolling(25, center=True, min_periods=3).median())
    return f


def speaker_levels(frames: pd.DataFrame, words: pd.DataFrame) -> dict:
    """Typical loudness (dB) and pitch (Hz) while the main speaker is talking."""
    f = _smooth(frames)
    m = np.zeros(len(f), bool)
    t = f.time.to_numpy()
    for s, e in words[["start", "end"]].itertuples(index=False):
        m |= (t >= s) & (t <= e)
    return {"loudness_db": float(np.nanmedian(f.loud_s[m])), "pitch_hz": float(np.nanmedian(f.pitch_s[m]))}


def _sample(f: pd.DataFrame, col: str, times: np.ndarray) -> np.ndarray:
    t = f.time.to_numpy()
    hop = t[1] - t[0]
    idx = np.round((times - t[0]) / hop).astype(int)
    out = np.full(len(times), np.nan)
    ok = (idx >= 0) & (idx < len(t))
    out[ok] = f[col].to_numpy()[idx[ok]]
    return out


def ending_stats(words: pd.DataFrame, frames: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """One row per sentence: how its ending was delivered.

    end_loudness_rel_db : loudness over the last 1 s of the sentence, vs. typical
    peak_loudness_rel_db: loudest point in the last 3 s, vs. typical
    end_pitch_rel_st    : pitch over the last 0.5 s, in semitones vs. typical
    pitch_fall_st       : pitch in the last 0.5 s minus pitch 1.5-0.5 s before (negative = falling)
    pace_wps_last5s     : words per second over the last 5 s of the sentence
    pause_after_s       : silence before the next word
    """
    f = _smooth(frames)
    rows = []
    for sid, s in words.groupby("seg_id"):
        end = s.end.max()

        def window(col, a, b):
            v = _sample(f, col, np.arange(end - a, end - b, 0.01))
            return np.nanmean(v) if np.isfinite(v).any() else np.nan

        p_end, p_before = window("pitch_s", 0.5, 0), window("pitch_s", 1.5, 0.5)
        last5 = words[(words.end > end - 5) & (words.end <= end)]
        rows.append({
            "seg_id": sid,
            "end_loudness_rel_db": window("loud_s", 1.0, 0) - levels["loudness_db"],
            "peak_loudness_rel_db": np.nanmax(_sample(f, "loud_s", np.arange(end - 3, end, 0.01)))
            - levels["loudness_db"],
            "end_pitch_rel_st": 12 * np.log2(p_end / levels["pitch_hz"]) if np.isfinite(p_end) else np.nan,
            "pitch_fall_st": 12 * np.log2(p_end / p_before) if np.isfinite(p_end) and np.isfinite(p_before)
            else np.nan,
            "pace_wps_last5s": len(last5) / min(5.0, end - s.start.min() + 1e-9) if len(last5) else np.nan,
            "pause_after_s": s.pause_after.iloc[-1],
        })
    return pd.DataFrame(rows)


def runup_curves(words: pd.DataFrame, frames: pd.DataFrame, levels: dict,
                 before_s: float = 6.0, after_s: float = 2.0) -> pd.DataFrame:
    """Long table: seg_id, rel_time, loudness_rel_db, pitch_rel_st for every sentence ending."""
    f = _smooth(frames)
    rel = np.round(np.arange(-before_s, after_s + 1e-9, HOP_S), 3)
    parts = []
    for sid, s in words.groupby("seg_id"):
        times = s.end.max() + rel
        loud = _sample(f, "loud_s", times) - levels["loudness_db"]
        pitch = 12 * np.log2(_sample(f, "pitch_s", times) / levels["pitch_hz"])
        parts.append(pd.DataFrame({"seg_id": sid, "rel_time": rel, "loudness_rel_db": loud, "pitch_rel_st": pitch}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def reaction_moments(events: pd.DataFrame, words: pd.DataFrame, sentences: pd.DataFrame,
                     window_s: float = 3.0) -> pd.DataFrame:
    """Every detected reaction, with what immediately preceded it.

    `mid_sentence` is True when the last word before the reaction is not the
    end of its sentence (the crowd broke in, as often happens with King).
    Reactions more than `window_s` after the last word keep their row but
    have no sentence attached.
    """
    w = words.sort_values("end").reset_index(drop=True)
    last_of_seg = w.groupby("seg_id").end.transform("max") == w.end
    rows = []
    for ev in events.itertuples(index=False):
        before = w[w.end <= ev.start + 0.25]
        row = {"label": ev.label, "start": ev.start, "end": ev.end, "duration": ev.duration, "peak": ev.peak}
        if len(before) and ev.start - before.end.iloc[-1] <= window_s:
            i = before.index[-1]
            row.update({"after_word": w.word[i], "delay_s": max(ev.start - w.end[i], 0.0),
                        "seg_id": w.seg_id[i], "mid_sentence": not bool(last_of_seg[i])})
        rows.append(row)
    rm = pd.DataFrame(rows)
    if "seg_id" in rm and len(sentences):
        rm = rm.merge(sentences, on="seg_id", how="left", suffixes=("", "_sentence"))
    return rm
