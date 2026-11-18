"""Pure-logic checks for the newer modules (no models, no network)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from speechart.aftermath import echo_summary
from speechart.reactions import ReactionConfig, reaction_events
from speechart.tagging import SpeechAnnotation, to_table
from speechart.timeline import clip_long_words, reaction_after


def test_reaction_events_merge_and_filter():
    t = np.arange(0, 10, 0.25)
    applause = np.zeros_like(t)
    applause[(t >= 2) & (t < 4)] = 0.8      # 2 s event
    applause[(t >= 4.25) & (t < 5)] = 0.8   # gap of 0.25 s -> merged with the first
    applause[(t >= 8) & (t < 8.5)] = 0.9    # 0.5 s blip -> dropped (< min_event_s)
    scores = pd.DataFrame({"time": t, "applause": applause, "laughter": 0.0, "cheering": 0.0})
    ev = reaction_events(scores, ReactionConfig())
    assert len(ev) == 1
    assert ev.start.item() == pytest.approx(1.875) and ev.end.item() == pytest.approx(4.875)


def test_clip_long_words_keeps_end():
    w = pd.DataFrame({"word": ["The", "idea"], "start": [130.4, 137.8], "end": [137.7, 138.2]})
    out = clip_long_words(w)
    assert out.end.tolist() == [137.7, 138.2]
    assert out.start[0] == pytest.approx(137.7 - (0.07 * 3 + 0.10))
    assert out.clipped.tolist() == [True, False]


def test_reaction_credited_to_last_word_before():
    w = pd.DataFrame({"word": ["how", "dare", "you?", "for"], "start": [0, 0.3, 0.6, 10],
                      "end": [0.25, 0.55, 0.9, 10.3]})
    ev = pd.DataFrame({"label": ["applause"], "start": [1.5], "end": [8.5], "duration": [7.0], "peak": [0.8]})
    out = reaction_after(w, ev)
    assert out.reaction_after.tolist() == ["", "", "applause", ""]
    assert out.reaction_delay[2] == pytest.approx(0.6)


def test_annotation_schema_and_table():
    ann = SpeechAnnotation(sentences=[{"seg_id": 8, "devices": ["RHETORICAL_QUESTION", "PUNCHLINE"],
                                       "emotion": "ANGER", "intensity": 3, "addressee": "LEADERS",
                                       "topic": "MORAL_JUDGEMENT", "invites_applause": True, "note": "x"}])
    assert to_table(ann).devices.item() == "RHETORICAL_QUESTION|PUNCHLINE"
    with pytest.raises(Exception):
        SpeechAnnotation(sentences=[{**ann.sentences[0].model_dump(), "intensity": 7}])


def test_echo_summary():
    days = pd.date_range("2019-09-09", "2019-09-30")
    views = np.where(days < "2019-09-23", 100, 100)
    views[days == "2019-09-24"] = 4000
    df = pd.DataFrame({"date": days, "article": "A", "views": views})
    e = echo_summary(df, date(2019, 9, 23)).iloc[0]
    assert e.baseline == 100 and e.peak == 4000 and e.days_after == 1 and e.x_baseline == 40


def test_find_phrase_ignores_case_and_punctuation():
    from speechart.art.charts import find_phrase

    w = pd.DataFrame({"word": ["How", "dare", "you?", "I", "say", "how", "dare", "you."]})
    assert find_phrase(w, "how dare you") == [0, 5]
