"""Synthetic checks: a 'voice' made of tone bursts with known gaps."""

import numpy as np
import pandas as pd
import pytest

from speechart.delivery import DeliveryConfig, detect_pauses, frame_features
from speechart.timeline import lagged_correlation, mark_intervals, resample, words_with_delivery

SR = 16_000
CFG = DeliveryConfig(sr=SR)

# (start, end, pitch Hz, amplitude) for each "word"; the gaps between are pauses
WORDS = [(0.30, 0.80, 220.0, 0.30), (0.90, 1.40, 220.0, 0.30),  # 0.10 s gap: too short to count
         (2.40, 3.00, 330.0, 0.60),                              # 1.00 s pause before
         (3.50, 4.00, 180.0, 0.15)]                              # 0.50 s pause before
TOTAL = 4.5


@pytest.fixture(scope="module")
def signal():
    rng = np.random.default_rng(0)
    t = np.arange(int(TOTAL * SR)) / SR
    y = 0.002 * rng.standard_normal(len(t))  # room noise
    for start, end, f0, amp in WORDS:
        m = (t >= start) & (t < end)
        y[m] += amp * np.sin(2 * np.pi * f0 * t[m])
    return y.astype(np.float32)


@pytest.fixture(scope="module")
def frames(signal):
    return frame_features(signal, CFG)


def test_pauses_found_at_known_gaps(frames):
    pauses = detect_pauses(frames, CFG)
    assert len(pauses) == 2
    assert pauses["start"].tolist() == pytest.approx([1.40, 3.00], abs=0.05)
    assert pauses["duration"].tolist() == pytest.approx([1.00, 0.50], abs=0.06)


def test_pitch_tracks_tones(frames):
    voiced = frames.dropna(subset=["pitch_hz"])
    at = lambda s, e: voiced[(voiced.time > s) & (voiced.time < e)]["pitch_hz"].median()
    assert at(2.5, 2.9) == pytest.approx(330, rel=0.03)
    assert at(3.6, 3.9) == pytest.approx(180, rel=0.03)


def test_words_get_delivery_features(frames):
    words = pd.DataFrame([(f"w{i}", s, e) for i, (s, e, _, _) in enumerate(WORDS)],
                         columns=["word", "start", "end"])
    w = words_with_delivery(words, frames)
    # the loud word is loudest, the quiet word quietest
    assert w["loudness_mean"].idxmax() == 2 and w["loudness_mean"].idxmin() == 3
    assert w["pause_after"].tolist()[:3] == pytest.approx([0.10, 1.00, 0.50])
    assert np.isnan(w["pause_after"].iloc[-1])


def test_mark_and_resample(frames):
    pauses = detect_pauses(frames, CFG)
    g = resample(mark_intervals(frames, pauses, "pause"), 0.5)
    assert g.loc[g.time == 2.0, "pause"].item() == pytest.approx(0.8, abs=0.05)
    assert g.loc[g.time == 0.5, "pause"].item() == 0


def test_lagged_correlation_finds_lead():
    a = pd.Series(np.r_[np.zeros(10), 1, np.zeros(30)])
    b = a.shift(5).fillna(0)  # b happens 5 steps after a
    r = lagged_correlation(a, b, max_lag=8)
    assert r.idxmax() == 5
