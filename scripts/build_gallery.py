"""Build the gallery: numbered copies of every image plus a running document describing each.

Usage:
    uv run python scripts/build_gallery.py

Writes gallery/NN_<name>.png and gallery/README.md. Run it again whenever the
art changes: numbering follows the CATALOG order below, images that do not
exist yet are skipped, and stale numbered images are removed first.

To add a new image: add an entry to CATALOG (and, for a new kind of piece, a
description to SERIES). Nothing else needs changing.
"""

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GALLERY = ROOT / "gallery"
SPEECHES = ["thunberg_2019_un", "thunberg_2018_cop24", "malala_2013_un", "mlk_1963_dream"]

# ------------------------------------------------------------------ what each kind of image is

MODELS = {
    "whisper": "Whisper large-v3-turbo (speech recognition) with wav2vec2 forced alignment, via WhisperX",
    "panns": "PANNs Cnn14 (sound-event recognition trained on AudioSet) for applause, laughter and cheering",
    "gemini": "Gemini 2.5 Flash (LLM), 5 independent blind runs per speech combined by vote, for rhetorical "
              "devices, emotion, intensity and applause prediction",
    "minilm": "all-MiniLM-L6-v2 (sentence-embedding model) to measure similarity of meaning between sentences",
    "pyin": "pYIN pitch tracking and RMS loudness (signal processing, not AI)",
    "novelty": "Foote novelty on a self-similarity matrix (from music analysis) to find sections",
}

SERIES = {
    "constellation": dict(
        title="The Hidden Wiring of Speech: constellation",
        shows="All four speeches as spirals on one sheet. Each spiral is one speech read from the center outward; "
              "each stroke is one breath (the words between two pauses), bent by the melody of that breath and as "
              "wide as it was loud. Colored threads join sentences that echo each other inside a speech; white "
              "threads join sentences that echo each other between speeches, so speakers who never cite each other "
              "are visibly wired together. Gold arcs mark applause.",
        data="breath groups and pitch/loudness contours (output/<id>/structure/breath_groups.csv, frames.csv); "
             "emotion tags (sentences.csv); meaning echoes within and across speeches "
             "(output/compare/structure/echo_edges.csv); applause events (reaction_events.csv)",
        made="Each breath's pitch is smoothed to a low-order curve (its melodic gesture) and drawn along an "
             "Archimedean spiral; echo threads are curves between the sentences' positions. Rendered in Python "
             "(matplotlib) by src/speechart/art/wiring.py.",
        models=["whisper", "pyin", "panns", "gemini", "minilm"]),
    "wiring": dict(
        title="The Hidden Wiring of Speech",
        shows="One speech as a spiral from the center outward. Each stroke is one breath group, its curve the "
              "melodic gesture of that breath (a rise, an arch, a fall) and its width the loudness. Pauses are gaps. "
              "Color is the emotion the AI hears. Threads through the interior join sentences the AI finds echoing "
              "each other in meaning, far apart in time. Gold arcs are applause.",
        data="breath_groups.csv, frames.csv, sentences.csv (emotion), reaction_events.csv (applause), "
             "echo_edges.csv (within-speech echoes)",
        made="src/speechart/art/wiring.py (matplotlib); echoes from sentence embeddings, gestures from a cubic fit "
             "to each breath's pitch.",
        models=["whisper", "pyin", "panns", "gemini", "minilm"]),
    "wiring_annotated": dict(
        title="The Hidden Wiring of Speech, annotated",
        shows="The same portrait with the speech's sections marked and labeled by the words that open them, plus "
              "a legend. The sections were found by a novelty detector that only sees changes in meaning and "
              "delivery, yet they fall on the speech's real rhetorical turns.",
        data="as the portrait, plus output/<id>/structure/sections.csv",
        made="src/speechart/art/wiring.py with annotated=True.",
        models=["whisper", "pyin", "panns", "gemini", "minilm", "novelty"]),
    "bodies": dict(
        title="Speech Bodies: all four speeches",
        shows="Each speech as a solid of stacked discs, all stretched to the same length so their shapes compare. "
              "Disc size is loudness, the spine bends with the melody of each breath, pauses pinch the body to a "
              "thread, color is the emotion the AI hears, and a faint gold body behind is the room applauding.",
        data="frames.csv (loudness, pitch), breath_groups.csv, sentences.csv (emotion), reaction_scores.csv "
             "(applause probability)",
        made="src/speechart/art/bodies.py (matplotlib ellipse collections, ~4500 discs per speech).",
        models=["pyin", "whisper", "gemini", "panns"]),
    "body": dict(
        title="Speech Body",
        shows="One speech as a solid of stacked discs, read left to right: disc size is loudness, the spine bends "
              "with the melody of each breath, pauses pinch the body, color is the emotion the AI hears, and the "
              "gold body behind is the room applauding.",
        data="frames.csv, breath_groups.csv, sentences.csv, reaction_scores.csv",
        made="src/speechart/art/bodies.py.",
        models=["pyin", "whisper", "gemini", "panns"]),
    "landscape": dict(
        title="Applause Landscape",
        shows="The speech as a mountain panorama read left to right. Gold peaks rise exactly where the audience "
              "responded; a paler far range follows loudness; ridge color is the emotion the AI hears; water at "
              "the base marks pauses.",
        data="timeline_100ms.csv (loudness, pauses, applause probability), words_timeline.csv (emotion)",
        made="src/speechart/art/landscape.py; also writes a 16-bit depth map for depth-conditioned AI painting.",
        models=["pyin", "panns", "gemini", "whisper"]),
    "ripples": dict(
        title="Ripples",
        shows="The speech at the center as a ring, read clockwise; rings spreading outward are the days after it, "
              "bright with attention to the speaker on Wikipedia; gold rays are news articles quoting the speech's "
              "key phrase, drawn at the angles where it was spoken so each echo traces back to its moment.",
        data="frames.csv, reaction_events.csv, aftermath_wikipedia.csv (Wikimedia pageviews), aftermath_gdelt.csv "
             "(GDELT news counts)",
        made="src/speechart/art/ripples.py. Aftermath data exists only from 2015 on, so only the Thunberg speeches "
             "have this piece.",
        models=["pyin", "panns", "whisper"]),
    "scroll": dict(
        title="Breath Scroll",
        shows="The speech as an ink scroll, cut into rows. Each word is a brush stroke whose weight follows "
              "loudness and whose height follows pitch; pauses are blank paper; gold leaf marks applause; red seals "
              "mark the sentences the AI predicted would draw applause.",
        data="frames.csv, words_timeline.csv, reaction_events.csv, sentences.csv (applause prediction)",
        made="src/speechart/art/breath_scroll.py.",
        models=["whisper", "pyin", "panns", "gemini"]),
    "fingerprint": dict(
        title="Speech Fingerprint",
        shows="A self-similarity matrix: time runs along both axes and a cell is bright when two moments resemble "
              "each other. Upper half: what is said (meaning); lower half: how it is said (loudness, pitch, "
              "silence). Repetition shows as threads, sections as blocks; gold on the diagonal is applause.",
        data="timeline_100ms.csv, words_timeline.csv",
        made="src/speechart/art/fingerprint.py.",
        models=["minilm", "whisper", "pyin", "panns"]),
    "phrase": dict(
        title="Phrase repetitions",
        shows="Every repetition of the speech's tracked phrase, overlaid and aligned on its first word: loudness, "
              "pitch and applause probability, so the deliveries can be compared like for like.",
        data="frames.csv, words_timeline.csv, reaction_scores.csv, reaction_events.csv",
        made="src/speechart/art/charts.py (analysis chart).",
        models=["whisper", "pyin", "panns"]),
    "anatomy": dict(
        title="Anatomy of applause",
        shows="Every sentence ending aligned on its last word, split by whether applause followed: loudness and "
              "pitch in the seconds before and after, single sentences thin, averages thick.",
        data="frames.csv, words_timeline.csv (output/<id>/runup_curves.csv holds the same curves)",
        made="src/speechart/art/charts.py (analysis chart).",
        models=["whisper", "pyin", "panns"]),
    "profiles": dict(
        title="Speech profiles", shows="Small multiples comparing pace, pauses, loudness and pitch range, applause "
        "rate, AI-rated intensity and how often predicted applause came.",
        data="output/compare/metrics.csv", made="src/speechart/compare.py (analysis chart).",
        models=["whisper", "pyin", "panns", "gemini"]),
    "devices": dict(
        title="Rhetorical devices per speech", shows="Share of sentences using each rhetorical device, per speech.",
        data="output/compare/metrics.csv (dev_* columns)", made="src/speechart/compare.py (analysis chart).",
        models=["gemini"]),
    "runup": dict(
        title="The run-up to applause, compared", shows="Loudness and pitch before applauded sentence endings, "
        "relative to each speaker's own typical level, one line per speech.",
        data="output/compare/runup_curves_all.csv", made="src/speechart/compare.py (analysis chart).",
        models=["whisper", "pyin", "panns"]),
    "echo": dict(
        title="The echo of each speech on Wikipedia", shows="Daily Wikipedia views of the speaker's article around "
        "each speech, aligned on the speech day (log scale).",
        data="aftermath_wikipedia.csv (Wikimedia pageviews)", made="src/speechart/compare.py (analysis chart).",
        models=[]),
    "fingerprints_sheet": dict(
        title="Speech Fingerprints side by side", shows="The four fingerprints on one sheet.",
        data="see Speech Fingerprint", made="src/speechart/compare.py (contact sheet).",
        models=["minilm", "whisper", "pyin", "panns"]),
    "landscapes_sheet": dict(
        title="Applause Landscapes stacked", shows="The four landscapes on one sheet.",
        data="see Applause Landscape", made="src/speechart/compare.py (contact sheet).",
        models=["pyin", "panns", "gemini", "whisper"]),
}

# ------------------------------------------------------------------ what goes in, in order


def catalog() -> list[dict]:
    """(series, speech or None, source path, short name) in gallery order."""
    out = ROOT / "output"
    c = [("constellation", None, out / "compare/art/constellation.png", "constellation"),
         ("bodies", None, out / "compare/art/speech_bodies.png", "speech_bodies")]
    per_speech = [("wiring", "art/wiring.png", "wiring"), ("body", "art/speech_body.png", "speech_body"),
                  ("landscape", "art/applause_landscape.png", "landscape"), ("ripples", "art/ripples.png", "ripples"),
                  ("scroll", "art/breath_scroll_preview.png", "breath_scroll"),
                  ("fingerprint", "art/speech_fingerprint.png", "fingerprint"),
                  ("wiring_annotated", "art/wiring_annotated.png", "wiring_annotated"),
                  ("phrase", "art/phrase_repetitions.png", "phrase_repetitions"),
                  ("anatomy", "art/anatomy_of_applause.png", "anatomy_of_applause")]
    for series, rel, name in per_speech:
        for sid in SPEECHES:
            c.append((series, sid, out / sid / rel, f"{name}_{sid}"))
    c += [("profiles", None, out / "compare/speech_profiles.png", "speech_profiles"),
          ("devices", None, out / "compare/rhetorical_devices.png", "rhetorical_devices"),
          ("runup", None, out / "compare/applause_runup.png", "applause_runup"),
          ("echo", None, out / "compare/echo_wikipedia.png", "echo_wikipedia"),
          ("fingerprints_sheet", None, out / "compare/fingerprints.png", "fingerprints_sheet"),
          ("landscapes_sheet", None, out / "compare/landscapes.png", "landscapes_sheet")]
    return [dict(series=s, sid=sid, src=p, name=n) for s, sid, p, n in c if p.exists()]


SECTIONS = [("Featured artworks", {"constellation", "bodies", "wiring", "body"}),
            ("Earlier artworks", {"landscape", "ripples", "scroll", "fingerprint"}),
            ("Annotated and analysis images", {"wiring_annotated", "phrase", "anatomy", "profiles", "devices",
                                               "runup", "echo", "fingerprints_sheet", "landscapes_sheet"})]


def speech_label(sid: str) -> str:
    """Short unambiguous name: speaker, event, year."""
    meta = json.loads((ROOT / "data" / "speeches" / sid / "meta.json").read_text())
    return f"{meta['speaker']}, {meta['title']} ({meta['date'][:4]})"


def speech_facts(sid: str, metrics: pd.DataFrame) -> str:
    meta = json.loads((ROOT / "data" / "speeches" / sid / "meta.json").read_text())
    who = f"{meta['speaker']}, {meta['title']}, {meta['venue']}, {meta['date']}"
    if meta.get("excerpt"):
        who += " (edited excerpt)"
    row = metrics[metrics.speech.str.startswith(meta["speaker"].split()[-1] if "King" not in meta["speaker"]
                                                else "King") & metrics.speech.str.contains(meta["date"][:4])]
    if row.empty:
        return who
    r = row.iloc[0]
    return (f"{who}. {r.minutes_speaking:.1f} min of speech, {int(r.sentences)} sentences, "
            f"{r.words_per_minute:.0f} words/min, {int(r.applause_events)} bursts of applause, "
            f"most common emotion (AI): {str(r.top_emotion).lower()}.")


def main() -> None:
    GALLERY.mkdir(exist_ok=True)
    for old in GALLERY.glob("[0-9][0-9]_*.png"):
        old.unlink()
    metrics = pd.read_csv(ROOT / "output" / "compare" / "metrics.csv")
    items = catalog()
    for i, it in enumerate(items, 1):
        it["n"] = i
        it["file"] = f"{i:02d}_{it['name']}.png"
        shutil.copy2(it["src"], GALLERY / it["file"])

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    L = ["# Gallery: Hidden Structures in Speech", "",
         f"Running document, regenerated {date.today().isoformat()} from commit `{commit.stdout.strip()}` by "
         "`scripts/build_gallery.py`. Every image in this folder is listed below under the same number and "
         "file name. Speeches: Greta Thunberg (UN Climate Action Summit 2019; COP24 2018), Malala Yousafzai "
         "(UN Youth Assembly 2013), Martin Luther King Jr. (I Have a Dream 1963, edited excerpt).", "",
         "## Models and methods used", ""]
    L += [f"- **{k}**: {v}" for k, v in MODELS.items()]
    L += ["", "All images are drawn by Python code (matplotlib) directly from the data; no image is generated by "
          "an image model. The AI models above produce the data the images are drawn from. The code was written "
          "with Claude (Anthropic) in Claude Code.", "", "## Index", ""]
    for i, it in enumerate(items, 1):
        s = SERIES[it["series"]]
        who = "" if it["sid"] is None else ": " + speech_label(it["sid"])
        L.append(f"{i}. [{it['file']}]({it['file']}): {s['title']}{who}")
    for heading, members in SECTIONS:
        group = [it for it in items if it["series"] in members]
        if not group:
            continue
        L += ["", f"## {heading}", ""]
        for it in group:
            s = SERIES[it["series"]]
            title = s["title"] + ("" if it["sid"] is None else ": " + speech_label(it["sid"]))
            L += [f"### Image {it['n']:02d}: {title}", "", f"![{title}]({it['file']})", "",
                  f"- **File:** `gallery/{it['file']}` (from `{it['src'].relative_to(ROOT)}`)"]
            if it["sid"]:
                L.append(f"- **Speech:** {speech_facts(it['sid'], metrics)}")
            L += [f"- **What it shows:** {s['shows']}", f"- **Data:** {s['data']}", f"- **How it's made:** {s['made']}"]
            if s["models"]:
                L.append("- **AI and methods:** " + "; ".join(MODELS[m] for m in s["models"]) + ".")
            L.append("")
    (GALLERY / "README.md").write_text("\n".join(L) + "\n")
    print(f"gallery: {len(items)} images + README.md in {GALLERY.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
