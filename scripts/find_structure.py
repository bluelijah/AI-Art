"""Find hidden structure in each speech and across speeches.

Usage:
    uv run python scripts/find_structure.py thunberg_2019_un thunberg_2018_cop24 malala_2013_un mlk_1963_dream

Each speech must have been through run_pipeline.py. Writes:

output/<id>/structure/
  lead_lag.csv, lead_lag_peaks.csv   which signals come before applause / pauses, and by how long
  sentence_features.csv              every sentence with derived features (position, pace, devices...)
  correlations.csv                   Spearman rho between every pair of sentence features
  arc.csv                            the speech's shape over normalised time
  sections.csv, novelty.csv          where the speech changes character
  breath_groups.csv                  the units between pauses, with gesture labels
  rhythm.csv                         loudness autocorrelation (the speech's pulse)

output/compare/structure/
  echo_edges.csv                     sentences that echo each other in meaning, within and across speeches
  gesture_centroids.csv              the shared vocabulary of melodic / loudness shapes
  gesture_share.csv                  how often each speaker uses each gesture
  correlations_pooled.csv            sentence-feature correlations across all speeches together
  lead_lag_peaks_all.csv             every speech's lead/lag peaks side by side
  findings.md                        the strongest results, in words
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from speechart import structure as st
from speechart.art import fingerprint
from speechart.compare import short_label

ROOT = Path(__file__).resolve().parents[1]
LEAD_LAG_PAIRS = [
    ("pause", "applause_prob"), ("loudness_db", "applause_prob"), ("pitch_hz", "applause_prob"),
    ("intensity", "applause_prob"), ("applause_vote", "applause_prob"), ("loudness_db", "pause"),
    ("intensity", "loudness_db"),
]


def per_speech(sid: str) -> dict:
    out = ROOT / "output" / sid
    sd = out / "structure"
    sd.mkdir(exist_ok=True)
    meta = json.loads((ROOT / "data" / "speeches" / sid / "meta.json").read_text())
    words = pd.read_csv(out / "words_timeline.csv")
    sentences = pd.read_csv(out / "sentences.csv")
    grid = pd.read_csv(out / "timeline_100ms.csv")
    frames = pd.read_csv(out / "frames.csv")
    levels = json.loads((out / "speaker_levels.json").read_text())

    # sentence tags onto the 100 ms grid, so tags can be lagged against sound
    for col in ["intensity", "applause_vote"]:
        if col in sentences:
            grid[col] = grid.seg_id.map(sentences.set_index("seg_id")[col])

    ll = st.lead_lag(grid, LEAD_LAG_PAIRS)
    ll.round(4).to_csv(sd / "lead_lag.csv", index=False)
    peaks = st.lead_lag_peaks(ll)
    peaks.round(3).to_csv(sd / "lead_lag_peaks.csv", index=False)

    feats = st.sentence_features(sentences, words)
    feats.round(4).to_csv(sd / "sentence_features.csv", index=False)
    corr = st.sentence_correlations(feats)
    corr.round(4).to_csv(sd / "correlations.csv", index=False)
    st.arc(feats).round(4).to_csv(sd / "arc.csv", index=False)

    # sections: self-similarity of meaning + delivery on a 1 s grid
    g = grid.copy()
    g["bin"] = (g.time // 1.0).astype(int)
    g1 = g.groupby("bin").mean(numeric_only=True).reset_index(drop=True)
    s_text = np.nan_to_num(fingerprint.text_similarity(fingerprint.text_windows(words, g1.time.to_numpy())))
    s_del = fingerprint.delivery_similarity(g1)
    dur = g1.time.iloc[-1] - g1.time.iloc[0]
    # aim for sections of roughly a minute (at least 20 s), scaled to the speech's length
    sec, nov = st.sections(g1.time.to_numpy(), (s_text + s_del) / 2, kernel_s=max(20.0, dur / 8),
                           min_gap_s=max(20.0, dur / 12), prominence=0.2)
    first_sentence = lambda a, b: next((r.text for r in sentences.itertuples() if a <= r.start < b), "")
    sec["opens_with"] = [first_sentence(r.start, r.end) for r in sec.itertuples()]
    sec.round(3).to_csv(sd / "sections.csv", index=False)
    pd.DataFrame({"time": g1.time, "novelty": nov}).round(4).to_csv(sd / "novelty.csv", index=False)

    bg, contours = st.breath_groups(words, frames, levels)
    st.rhythm_periodicity(grid).round(4).to_csv(sd / "rhythm.csv", index=False)

    return {"sid": sid, "label": short_label(meta), "sentences": feats, "peaks": peaks, "corr": corr,
            "sections": sec, "breath_groups": bg, "contours": contours, "sd": sd}


def main(ids: list[str]) -> None:
    out = ROOT / "output" / "compare" / "structure"
    out.mkdir(parents=True, exist_ok=True)
    res = [per_speech(s) for s in ids]
    for r in res:
        print(f"{r['label']}: {len(r['sections'])} sections, {len(r['breath_groups'])} breath groups")

    # gestures: one shared vocabulary across all speakers
    all_contours = np.vstack([r["contours"] for r in res if len(r["contours"])])
    labels, centroids = st.gesture_vocabulary(all_contours, k=6)
    i = 0
    shares = []
    for r in res:
        bg = r["breath_groups"]
        bg["gesture"] = -1
        n = int(bg.has_contour.sum())
        bg.loc[bg.has_contour, "gesture"] = labels[i:i + n]
        i += n
        bg.round(3).to_csv(r["sd"] / "breath_groups.csv", index=False)
        share = bg[bg.gesture >= 0].gesture.value_counts(normalize=True).reindex(range(6), fill_value=0)
        shares.append(pd.Series(share.values, index=[f"gesture_{g}" for g in range(6)], name=r["label"]))
    pts = centroids.shape[1] // 2
    pd.DataFrame([{"gesture": g, "part": part, "point": p, "value": v}
                  for g in range(len(centroids)) for part, seg in (("pitch_st", centroids[g, :pts]),
                                                                   ("loudness_scaled", centroids[g, pts:]))
                  for p, v in enumerate(seg)]).round(4).to_csv(out / "gesture_centroids.csv", index=False)
    gshare = pd.DataFrame(shares)
    gshare.round(3).to_csv(out / "gesture_share.csv")

    # echoes: AI meaning links between sentences, within and across speeches
    items = pd.concat([r["sentences"].assign(speech_id=r["sid"])[["speech_id", "seg_id", "text", "start"]]
                       for r in res], ignore_index=True)
    edges = st.echo_edges(items, st.embed(items.text.tolist()))
    edges.round(4).to_csv(out / "echo_edges.csv", index=False)

    # pooled correlations and lead/lag side by side
    pooled = pd.concat([r["sentences"].assign(speech=r["label"]) for r in res], ignore_index=True)
    pc = st.sentence_correlations(pooled)
    pc.round(4).to_csv(out / "correlations_pooled.csv", index=False)
    lla = pd.concat([r["peaks"].assign(speech=r["label"]) for r in res], ignore_index=True)
    lla.round(3).to_csv(out / "lead_lag_peaks_all.csv", index=False)

    write_findings(res, pc, lla, edges, gshare, out / "findings.md")
    print(f"\nwrote {out.relative_to(ROOT)}/ and output/<id>/structure/")


def write_findings(res, pooled_corr, lla, edges, gshare, path: Path) -> None:
    L = ["# Structures found (auto-generated)", "",
         "Numbers to check, not conclusions. Small samples: read p-values and n.", ""]
    L += ["## Strongest sentence-level correlations (all speeches pooled, p < 0.05)", ""]
    sig = pooled_corr[(pooled_corr.p < 0.05) & ~pooled_corr.trivial]
    for r in sig.head(20).itertuples():
        L.append(f"- {r.a} ~ {r.b}: rho {r.rho:+.2f} (p {r.p:.3f}, n {r.n})")
    L += ["", "## Lead/lag peaks (positive lag = first signal comes before the second)", ""]
    for r in lla.sort_values(["a", "b", "speech"]).itertuples():
        flag = "  (edge of window: a trend, not a lead/lag)" if r.at_window_edge else ""
        L.append(f"- {r.speech}: {r.a} -> {r.b}: r {r.peak_r:+.2f} at {r.peak_lag_s:+.1f} s{flag}")
    L += ["", "## Sections per speech", ""]
    for r in res:
        L.append(f"- {r['label']}: {len(r['sections'])} sections; boundaries at "
                 + ", ".join(f"{s:.0f}s" for s in r["sections"].start.iloc[1:]))
    L += ["", "## Strongest meaning echoes across speeches", ""]
    for r in edges[edges.kind == "across"].head(10).itertuples():
        L.append(f"- {r.similarity:.2f}: [{r.speech_a}] \"{r.text_a}\" <-> [{r.speech_b}] \"{r.text_b}\"")
    L += ["", "## Strongest echoes within a speech (far apart in time)", ""]
    far = edges[(edges.kind == "within") & (edges.time_gap_s > 60)]
    for r in far.head(10).itertuples():
        L.append(f"- {r.similarity:.2f}, {r.time_gap_s:.0f}s apart [{r.speech_a}]: \"{r.text_a}\" <-> \"{r.text_b}\"")
    L += ["", "## Per-speech correlations (p < 0.05, non-trivial)", ""]
    for r in res:
        c = r["corr"][(r["corr"].p < 0.05) & ~r["corr"].trivial]
        L.append(f"- **{r['label']}**: " + "; ".join(f"{x.a} ~ {x.b} {x.rho:+.2f}" for x in c.head(8).itertuples()))
    L += ["", "## Gesture vocabulary share per speaker", "",
          "Gesture 0 is the flat shape (pitch moves < 1 semitone). Old or noisy recordings (King) read as flat",
          "more often because the pitch tracker loses the voice.", "", "```\n" + gshare.round(2).to_string() + "\n```", ""]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["thunberg_2019_un", "thunberg_2018_cop24", "malala_2013_un", "mlk_1963_dream"])
