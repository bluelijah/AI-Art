"""Checks for the per-item process tables (synthetic data, no models)."""

import numpy as np
import pandas as pd
import pytest

from speechart.process import ending_stats, reaction_moments, runup_curves, speaker_levels


@pytest.fixture
def frames():
    t = np.round(np.arange(0, 20, 0.01), 2)
    loud = np.full(len(t), -30.0)
    pitch = np.full(len(t), 200.0)
    pitch[(t >= 9.5) & (t < 10)] = 100.0  # sentence 1 ends an octave low
    return pd.DataFrame({"time": t, "loudness_db": loud, "pitch_hz": pitch, "voiced_prob": 0.9})


@pytest.fixture
def words():
    return pd.DataFrame({"word": ["we", "will", "win.", "and", "then"],
                         "start": [8.0, 8.6, 9.2, 12.0, 12.5], "end": [8.5, 9.1, 10.0, 12.4, 13.0],
                         "seg_id": [1, 1, 1, 2, 2], "pause_after": [0.1, 0.1, 2.0, 0.1, np.nan]})


def test_levels_and_pitch_fall(frames, words):
    lv = speaker_levels(frames, words)
    assert lv["loudness_db"] == pytest.approx(-30)
    stats = ending_stats(words, frames, lv).set_index("seg_id")
    assert stats.loc[1, "end_pitch_rel_st"] < -6          # well below typical pitch
    assert stats.loc[1, "pitch_fall_st"] < -6            # and falling into the end
    assert stats.loc[2, "end_pitch_rel_st"] == pytest.approx(0, abs=0.5)
    assert stats.loc[1, "pause_after_s"] == 2.0


def test_runup_curves_shape(frames, words):
    c = runup_curves(words, frames, speaker_levels(frames, words))
    assert set(c.seg_id) == {1, 2}
    assert c.rel_time.min() == pytest.approx(-6) and c.rel_time.max() == pytest.approx(2)


def test_reaction_moments_flags_mid_sentence(words):
    events = pd.DataFrame({"label": ["applause", "applause"], "start": [10.5, 12.5],
                           "end": [12.0, 14.0], "duration": [1.5, 1.5], "peak": [0.8, 0.7]})
    sentences = pd.DataFrame({"seg_id": [1, 2], "text": ["We will win.", "And then"]})
    m = reaction_moments(events, words, sentences)
    assert m.after_word.tolist() == ["win.", "and"]
    assert m.mid_sentence.tolist() == [False, True]
    assert m.text.tolist() == ["We will win.", "And then"]
