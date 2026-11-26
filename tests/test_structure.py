"""Checks for the structure-finding functions (synthetic data, no models)."""

import numpy as np
import pandas as pd

from speechart import structure as st


def test_novelty_finds_a_section_change():
    # two blocks: first half similar to itself, second half similar to itself
    n = 60
    ssm = np.zeros((n, n))
    ssm[:30, :30] = ssm[30:, 30:] = 1
    sec, nov = st.sections(np.arange(n, dtype=float), ssm, kernel_s=10, min_gap_s=10, prominence=0.2)
    assert len(sec) == 2 and abs(sec.start.iloc[1] - 30) <= 1
    assert nov.argmax() in range(28, 33)


def test_lead_lag_peak_and_edge_flag():
    a = np.zeros(300)
    a[[50, 150, 250]] = 1
    grid = pd.DataFrame({"x": a, "y": np.roll(a, 20), "trend": np.linspace(0, 1, 300)})  # y follows x by 2 s
    peaks = st.lead_lag_peaks(st.lead_lag(grid, [("x", "y"), ("trend", "x")], max_lag_s=4))
    xy = peaks[(peaks.a == "x") & (peaks.b == "y")].iloc[0]
    assert abs(xy.peak_lag_s - 2.0) < 1e-9 and not xy.at_window_edge


def test_trivial_pairs_are_flagged():
    s = pd.DataFrame({"duration_s": np.arange(20.0), "n_words": np.arange(20) * 3,
                      "intensity": np.tile([0, 1, 2, 3], 5), "got_applause": [0, 1] * 10,
                      "pause_after_s": np.arange(20.0)})
    c = st.sentence_correlations(s)
    flagged = c[c.trivial].apply(lambda r: {r.a, r.b}, axis=1).tolist()
    assert {"duration_s", "n_words"} in flagged and {"pause_after_s", "got_applause"} in flagged


def test_echo_edges_skip_neighbours_and_short_sentences():
    items = pd.DataFrame({"speech_id": ["a", "a", "a", "b"], "seg_id": [0, 1, 2, 0],
                          "text": ["we must act on climate now", "we must act on climate now", "Thank you.",
                                   "we have to act on climate today"],
                          "start": [0, 5, 10, 0]})
    emb = np.array([[1, 0], [1, 0], [1, 0], [0.9, 0.1]], float)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    e = st.echo_edges(items, emb, min_gap_items=2)
    assert not ((e.speech_a == "a") & (e.seg_a == 0) & (e.speech_b == "a") & (e.seg_b == 1)).any()  # neighbours
    assert not (e.text_a.eq("Thank you.") | e.text_b.eq("Thank you.")).any()                         # too short
    assert (e.kind == "across").any()


def test_gesture_vocabulary_separates_shapes():
    rise = np.r_[np.linspace(-3, 3, 16), np.zeros(16)]
    fall = -rise
    X = np.vstack([rise + 0.05 * i for i in range(10)] + [fall + 0.05 * i for i in range(6)])
    labels, cent = st.gesture_vocabulary(X, k=2)
    assert len(set(labels[:10])) == 1 and len(set(labels[10:])) == 1 and labels[0] != labels[10]
    assert labels[0] == 0  # most common gesture gets id 0


def test_spiral_runs_centre_outward_and_gesture_is_smooth():
    from speechart.art.wiring import Spiral, _gesture

    sp = Spiral(0, 100, turns=3, r0=0.2, r1=1.0)
    r = lambda t: np.hypot(*sp.point(t))
    assert r(0) == np.float64(0.2) or abs(r(0) - 0.2) < 1e-9
    assert abs(r(100) - 1.0) < 1e-9 and r(50) < r(80)
    t = np.linspace(0, 2, 200)
    noisy = 4 * np.sin(np.pi * t / 2) + np.sin(40 * t)  # an arch plus syllable wiggle
    g = _gesture(t, noisy)
    assert np.abs(np.diff(g, 2)).max() < 0.05      # no wiggle left
    assert g.argmax() in range(60, 140)            # the arch peaks mid-line
