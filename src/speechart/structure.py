"""Structure finding: patterns in the captured data that nobody hears directly.

Each function takes the per-speech tables the pipeline already writes and
returns a table, so every structure found can be inspected row by row and
then drawn. Grouped by the kind of structure:

Time and causality
- lead_lag            : does signal A tend to come before signal B, and by how long?
- sentence_correlations: which delivery features and AI tags move together?

Shape of the whole speech
- arc                 : build-up and release over the speech (smoothed intensity, loudness, pace)
- sections            : where the speech changes character (novelty peaks in self-similarity)

Rhythm
- breath_groups       : the units between pauses (the speech's "lines"), their length and shape
- rhythm_periodicity  : whether loudness pulses at a regular period (the speech's meter)

Hidden relationships
- echo_edges          : sentences that echo each other in meaning (AI embeddings), within and across speeches
- gesture_vocabulary  : recurring melodic/loudness shapes of breath groups, clustered across speakers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal, stats

from speechart.timeline import lagged_correlation

# ---------------------------------------------------------------- time and causality


def lead_lag(grid: pd.DataFrame, pairs: list[tuple[str, str]], step_s: float = 0.1,
             max_lag_s: float = 6.0) -> pd.DataFrame:
    """Correlation of a(t) with b(t + lag) for each pair, over lags in [-max, +max] seconds.

    Positive lag with a strong r = `a` tends to come before `b` by that much.
    Returns long format: a, b, lag_s, r.
    """
    max_lag = int(round(max_lag_s / step_s))
    rows = []
    for a, b in pairs:
        if a not in grid or b not in grid:
            continue
        r = lagged_correlation(grid[a], grid[b], max_lag)
        rows += [(a, b, lag * step_s, val) for lag, val in r.items()]
    return pd.DataFrame(rows, columns=["a", "b", "lag_s", "r"])


def lead_lag_peaks(ll: pd.DataFrame) -> pd.DataFrame:
    """Strongest correlation (by absolute value) for each pair, and the lag it occurs at."""
    idx = ll.groupby(["a", "b"]).r.apply(lambda s: s.abs().idxmax())
    out = ll.loc[idx.values].rename(columns={"lag_s": "peak_lag_s", "r": "peak_r"}).reset_index(drop=True)
    edge = ll.lag_s.abs().max()
    # a "peak" at the edge of the search window is a trend, not a real lead/lag
    out["at_window_edge"] = out.peak_lag_s.abs() >= edge - 1e-9
    return out


NUMERIC_SENTENCE_FEATURES = [
    "position", "duration_s", "n_words", "words_per_s", "end_loudness_rel_db", "peak_loudness_rel_db",
    "end_pitch_rel_st", "pitch_fall_st", "pace_wps_last5s", "pause_after_s", "intensity", "applause_vote",
    "emotion_agreement", "device_count", "got_applause",
]


# pairs related by definition or by measurement, not by anything the speaker did
TRIVIAL_PAIRS = {
    "duration_s~n_words", "duration_s~words_per_s", "n_words~words_per_s", "words_per_s~pace_wps_last5s",
    "duration_s~pace_wps_last5s", "n_words~pace_wps_last5s", "end_pitch_rel_st~pitch_fall_st",
    "end_loudness_rel_db~peak_loudness_rel_db",
    # applause fills the gap after a sentence, so the "pause" after it is long by construction
    "pause_after_s~got_applause",
}


def sentence_features(sentences: pd.DataFrame, words: pd.DataFrame) -> pd.DataFrame:
    """Add derived per-sentence features used for correlation and clustering."""
    s = sentences.copy()
    span0, span1 = words.start.min(), words.end.max()
    s["position"] = (s.start - span0) / (span1 - span0)
    s["duration_s"] = s.end - s.start
    s["n_words"] = s.seg_id.map(words.groupby("seg_id").size())
    s["words_per_s"] = s.n_words / s.duration_s.clip(lower=0.2)
    s["got_applause"] = (s.reaction_after.fillna("") == "applause").astype(int)
    if "devices" in s:
        devs = s.devices.fillna("NONE").str.split("|")
        s["device_count"] = devs.apply(lambda d: len([x for x in d if x != "NONE"]))
        for d in sorted({x for ds in devs for x in ds} - {"NONE"}):
            s[f"has_{d.lower()}"] = devs.apply(lambda ds, d=d: int(d in ds))
    return s


def sentence_correlations(s: pd.DataFrame, min_n: int = 8) -> pd.DataFrame:
    """Spearman correlation between every pair of numeric sentence features.

    Long format with rho, p-value and n, sorted by strength. Rank correlation,
    so it captures any monotonic relation and is robust to outliers.
    """
    cols = [c for c in NUMERIC_SENTENCE_FEATURES if c in s] + [c for c in s if c.startswith("has_")]
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            d = s[[a, b]].dropna()
            if len(d) < min_n or d[a].nunique() < 2 or d[b].nunique() < 2:
                continue
            rho, p = stats.spearmanr(d[a], d[b])
            rows.append((a, b, rho, p, len(d)))
    out = pd.DataFrame(rows, columns=["a", "b", "rho", "p", "n"])
    pair = out.a + "~" + out.b
    out["trivial"] = pair.isin(TRIVIAL_PAIRS) | pair.isin([f"{b}~{a}" for a_b in TRIVIAL_PAIRS
                                                          for a, b in [a_b.split("~")]])
    out.loc[out.a.str.startswith("has_") & (out.b == "device_count"), "trivial"] = True
    out.loc[out.b.str.startswith("has_") & (out.a == "device_count"), "trivial"] = True
    return out.reindex(out.rho.abs().sort_values(ascending=False).index).reset_index(drop=True)


# ---------------------------------------------------------------- shape of the whole speech


def arc(s: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """The speech's shape over normalised time (0 = start, 1 = end), in `bins` slices.

    For each slice: mean AI intensity, loudness at sentence ends, pace, share of
    sentences with applause, and share predicted to invite applause.
    """
    s = s.copy()
    s["slice"] = np.minimum((s.position * bins).astype(int), bins - 1)
    agg = {"intensity": "mean", "peak_loudness_rel_db": "mean", "words_per_s": "mean", "got_applause": "mean"}
    if "applause_vote" in s:
        agg["applause_vote"] = "mean"
    a = s.groupby("slice").agg({k: v for k, v in agg.items() if k in s}).reindex(range(bins))
    a.insert(0, "position", (np.arange(bins) + 0.5) / bins)
    return a.reset_index(drop=True)


def _checkerboard(n: int) -> np.ndarray:
    g = signal.windows.gaussian(2 * n, std=n / 2)
    g2 = np.outer(g, g)
    sign = np.ones((2 * n, 2 * n))
    sign[:n, n:] = sign[n:, :n] = -1
    return g2 * sign


def novelty(ssm: np.ndarray, kernel_half: int) -> np.ndarray:
    """Foote novelty: slide a checkerboard kernel down the diagonal of a self-similarity matrix.

    Peaks mark moments where what came before is unlike what comes after:
    section boundaries, the way music analysis finds verse/chorus changes.
    """
    k = _checkerboard(kernel_half)
    n = len(ssm)
    pad = np.pad(np.nan_to_num(ssm), kernel_half, mode="constant")
    nov = np.array([(pad[i:i + 2 * kernel_half, i:i + 2 * kernel_half] * k).sum() for i in range(n)])
    nov -= nov.min()
    return nov / nov.max() if nov.max() > 0 else nov


def sections(times: np.ndarray, ssm: np.ndarray, kernel_s: float = 12.0, step_s: float = 1.0,
             min_gap_s: float = 15.0, prominence: float = 0.15) -> tuple[pd.DataFrame, np.ndarray]:
    """Split the speech where its self-similarity changes. Returns (sections table, novelty curve)."""
    nov = novelty(ssm, max(2, int(kernel_s / step_s / 2)))
    peaks, _ = signal.find_peaks(nov, distance=max(1, int(min_gap_s / step_s)), prominence=prominence)
    bounds = [times[0]] + [times[p] for p in peaks] + [times[-1] + step_s]
    sec = pd.DataFrame({"section": range(len(bounds) - 1), "start": bounds[:-1], "end": bounds[1:]})
    sec["duration_s"] = sec.end - sec.start
    sec["boundary_strength"] = [np.nan] + [float(nov[p]) for p in peaks]
    return sec, nov


# ---------------------------------------------------------------- rhythm


def breath_groups(words: pd.DataFrame, frames: pd.DataFrame, levels: dict, min_pause_s: float = 0.3,
                  contour_points: int = 16) -> tuple[pd.DataFrame, np.ndarray]:
    """Split the speech into breath groups: runs of words between pauses of at least `min_pause_s`.

    These are the speech's natural "lines", like lines of verse. Returns a table
    (one row per group) and an array of normalised contours (pitch and loudness,
    resampled to `contour_points` each) for clustering into gestures.
    """
    w = words.sort_values("start").reset_index(drop=True)
    gap = w.start.shift(-1) - w.end
    group = np.r_[0, np.cumsum((gap >= min_pause_s).to_numpy()[:-1])]
    w["group"] = group

    t = frames.time.to_numpy()
    loud = frames.loudness_db.rolling(9, center=True, min_periods=1).mean().to_numpy()
    pitch = (frames.pitch_hz.where(frames.voiced_prob > 0.1)
             .rolling(15, center=True, min_periods=3).median().to_numpy())

    rows, contours = [], []
    for g, ws in w.groupby("group"):
        s, e = ws.start.min(), ws.end.max()
        m = (t >= s) & (t <= e)
        lp = loud[m] - levels["loudness_db"]
        pp = 12 * np.log2(pitch[m] / levels["pitch_hz"])
        ok = np.isfinite(pp)
        x = np.linspace(0, 1, contour_points)
        if m.sum() >= 4 and ok.sum() >= 4:
            xp = np.linspace(0, 1, m.sum())
            pc = np.interp(x, xp[ok], pp[ok])
            lc = np.interp(x, xp, lp)
            contours.append(np.r_[pc - pc.mean(), (lc - lc.mean()) / 6.0])  # shape only, comparable scales
            has_contour = True
        else:
            has_contour = False
        rows.append({"group": g, "start": s, "end": e, "duration_s": e - s, "n_words": len(ws),
                     "text": " ".join(ws.word), "seg_ids": "|".join(map(str, sorted(ws.seg_id.unique()))),
                     "pause_after_s": float(gap.loc[ws.index[-1]]) if pd.notna(gap.loc[ws.index[-1]]) else np.nan,
                     "mean_loudness_rel_db": float(np.nanmean(lp)) if m.any() else np.nan,
                     "mean_pitch_rel_st": float(np.nanmean(pp)) if ok.any() else np.nan,
                     "has_contour": has_contour})
    return pd.DataFrame(rows), np.array(contours)


def rhythm_periodicity(grid: pd.DataFrame, step_s: float = 0.1, min_s: float = 1.0,
                       max_s: float = 12.0) -> pd.DataFrame:
    """Autocorrelation of the loudness envelope: a regular pulse shows as a peak at its period."""
    x = grid.loudness_db.fillna(grid.loudness_db.min()).to_numpy()
    x = (x - x.mean()) / (x.std() or 1)
    ac = np.correlate(x, x, mode="full")[len(x) - 1:] / len(x)
    lags = np.arange(len(ac)) * step_s
    m = (lags >= min_s) & (lags <= max_s)
    return pd.DataFrame({"lag_s": lags[m], "autocorr": ac[m]})


# ---------------------------------------------------------------- hidden relationships


def embed(texts: list[str]) -> np.ndarray:
    from speechart.art.fingerprint import EMBED_MODEL
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return model.encode(texts, normalize_embeddings=True, batch_size=64)


def echo_edges(items: pd.DataFrame, emb: np.ndarray, top_k: int = 3, min_sim: float = 0.45,
               min_gap_items: int = 2, min_words: int = 4) -> pd.DataFrame:
    """Meaning-echo links between sentences (AI embeddings, cosine similarity).

    `items` needs columns speech_id, seg_id, text, start. Each sentence keeps its
    `top_k` most similar partners above `min_sim`, skipping immediate neighbours
    within the same speech (those are trivially similar). Edges are undirected
    and deduplicated; `kind` says whether the echo stays within one speech or
    crosses between speeches.
    """
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1)
    it = items.reset_index(drop=True)
    same = it.speech_id.to_numpy()[:, None] == it.speech_id.to_numpy()[None, :]
    order = it.groupby("speech_id").cumcount().to_numpy()
    near = same & (np.abs(order[:, None] - order[None, :]) < min_gap_items)
    sim = np.where(near, -1, sim)
    short = it.text.str.split().str.len().to_numpy() < min_words  # "Thank you." echoes everything
    sim[short, :] = -1
    sim[:, short] = -1
    edges = set()
    for i in range(len(it)):
        for j in np.argsort(sim[i])[::-1][:top_k]:
            if sim[i, j] >= min_sim:
                edges.add((min(i, j), max(i, j)))
    rows = []
    for i, j in sorted(edges):
        a, b = it.iloc[i], it.iloc[j]
        rows.append({"speech_a": a.speech_id, "seg_a": a.seg_id, "text_a": a.text,
                     "speech_b": b.speech_id, "seg_b": b.seg_id, "text_b": b.text,
                     "similarity": float(sim[i, j]),
                     "kind": "within" if a.speech_id == b.speech_id else "across",
                     "time_gap_s": float(abs(b.start - a.start)) if a.speech_id == b.speech_id else np.nan})
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).reset_index(drop=True)


def gesture_vocabulary(contours: np.ndarray, k: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Cluster breath-group contours into `k` recurring gestures (k-means).

    Returns (labels, centroids). Each centroid is a typical melodic + loudness
    shape: e.g. a rise-fall arch, a steady fall, a flat drive. The share of
    each gesture per speaker is that speaker's prosodic vocabulary.
    """
    from scipy.cluster.vq import kmeans2

    rng = np.random.default_rng(seed)
    X = np.nan_to_num(contours)
    best = None
    for _ in range(10):  # several restarts, keep the tightest clustering
        cent, lab = kmeans2(X, k, minit="++", seed=rng)
        inertia = ((X - cent[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab, cent)
    # order gestures by how common they are, so ids are stable and meaningful
    counts = np.bincount(best[1], minlength=k)
    order = np.argsort(-counts)
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    return remap[best[1]], best[2][order]
