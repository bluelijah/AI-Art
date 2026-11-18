"""Cross-speech comparison: one row of comparable numbers per speech.

Everything is computed over the main speaker's span only (from
meta.speaker_start_s to the last word, plus any reaction that follows it),
and normalised per minute or as shares, so speeches of different lengths
can sit side by side.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from speechart.art.charts import GRID, INK, INK_2, SERIES, SURFACE, _style

ROOT = Path(__file__).resolve().parents[2]
DEVICES = ["CONTRAST", "LIST_OF_THREE", "ANAPHORA", "RHETORICAL_QUESTION", "DIRECT_ADDRESS", "STATISTIC",
           "PERSONAL_TESTIMONY", "ACCUSATION", "WARNING", "PUNCHLINE"]


def load(speech_id: str) -> dict:
    meta = json.loads((ROOT / "data" / "speeches" / speech_id / "meta.json").read_text())
    out = ROOT / "output" / speech_id
    d = {"id": speech_id, "meta": meta, "out": out}
    for name in ["frames", "pauses", "words_timeline", "reaction_events", "timeline_100ms"]:
        p = out / f"{name}.csv"
        d[name] = pd.read_csv(p) if p.exists() else None
    return d


def short_label(meta: dict) -> str:
    name = meta["speaker"].split()[-1] if "King" not in meta["speaker"] else "King"
    return f"{name} {meta['date'][:4]}" + (" (excerpt)" if meta.get("excerpt") else "")


def speech_metrics(d: dict) -> dict:
    w, f, p, ev = d["words_timeline"], d["frames"], d["pauses"], d["reaction_events"]
    t0, t_end = w.start.min(), w.end.max()
    minutes = (t_end - t0) / 60
    f = f[(f.time >= t0) & (f.time <= t_end)]
    p = p[(p.start >= t0) & (p.end <= t_end)]
    ev = ev[(ev.end > t0)]
    app = ev[ev.label == "applause"]
    sent = w.groupby("seg_id").first()
    voiced = f.pitch_hz.where(f.voiced_prob > 0.5).dropna()

    m = {
        "speech": short_label(d["meta"]),
        "minutes_speaking": minutes,
        "words_per_minute": len(w) / minutes,
        "sentences": len(sent),
        "words_per_sentence": len(w) / len(sent),
        "pauses_per_minute": len(p) / minutes,
        "median_pause_s": p.duration.median(),
        "share_time_paused": p.duration.sum() / (t_end - t0),
        "loudness_range_db": np.percentile(f.loudness_db, 90) - np.percentile(f.loudness_db, 10),
        "pitch_median_hz": voiced.median(),
        "pitch_range_semitones": 12 * np.log2(np.percentile(voiced, 95) / np.percentile(voiced, 5)),
        "applause_events": len(app),
        "applause_per_minute": len(app) / minutes,
        "applause_seconds": app.duration.sum(),
        "laughter_events": int((ev.label == "laughter").sum()),
        "cheering_events": int((ev.label == "cheering").sum()),
    }
    if "invites_applause" in sent:
        got = w.groupby("seg_id").reaction_after.apply(lambda x: (x == "applause").any())
        inv = sent.invites_applause.astype(bool)
        m["predicted_applause_sentences"] = int(inv.sum())
        m["prediction_hit_rate"] = (inv & got).sum() / max(inv.sum(), 1)   # precision
        m["applause_caught"] = (inv & got).sum() / max(got.sum(), 1)       # recall
        m["mean_intensity"] = sent.intensity.mean()
        devs = sent.devices.fillna("").str.split("|")
        for dv in DEVICES:
            m[f"dev_{dv.lower()}"] = devs.apply(lambda x: dv in x).mean()
        m["top_emotion"] = sent.emotion.mode().iloc[0]
    return m


def metrics_table(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame([speech_metrics(load(i)) for i in ids])


def profile_chart(table: pd.DataFrame, path: str) -> None:
    """Small multiples: one panel per measure, one bar per speech, fixed colour per speech."""
    panels = [
        ("words_per_minute", "words per minute", "{:.0f}"),
        ("pauses_per_minute", "pauses per minute", "{:.1f}"),
        ("share_time_paused", "share of time in pauses", "{:.0%}"),
        ("median_pause_s", "median pause (s)", "{:.2f}"),
        ("loudness_range_db", "loudness range (dB, p10-p90)", "{:.0f}"),
        ("pitch_range_semitones", "pitch range (semitones, p5-p95)", "{:.1f}"),
        ("applause_per_minute", "applause bursts per minute", "{:.2f}"),
        ("mean_intensity", "mean emotional intensity (0-3)", "{:.1f}"),
        ("prediction_hit_rate", "predicted applause that came", "{:.0%}"),
    ]
    panels = [pn for pn in panels if pn[0] in table]
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.0 * rows + 0.8), dpi=150, facecolor=SURFACE)
    colours = {s: SERIES[i] for i, s in enumerate(table.speech)}
    y = np.arange(len(table))
    for ax, (col, label, fmt) in zip(axes.flat, panels):
        vals = table[col].to_numpy(dtype=float)
        ax.barh(y, vals, color=[colours[s] for s in table.speech], height=0.62)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8, color=INK)
        ax.set_yticks(y, table.speech, fontsize=8, color=INK_2)
        ax.invert_yaxis()
        _style(ax, "")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xticks([])
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        ax.set_xlim(0, np.nanmax(vals) * 1.3 if np.isfinite(np.nanmax(vals)) and np.nanmax(vals) > 0 else 1)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("Speech profiles", x=0.02, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def device_chart(table: pd.DataFrame, path: str) -> None:
    """Share of sentences using each rhetorical device, per speech (dot plot)."""
    cols = [c for c in table if c.startswith("dev_")]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150, facecolor=SURFACE)
    y = np.arange(len(cols))
    for i, row in table.iterrows():
        ax.scatter(row[cols].astype(float), y + (i - (len(table) - 1) / 2) * 0.16, s=36, color=SERIES[i],
                   label=row.speech, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    ax.set_yticks(y, [c[4:].replace("_", " ") for c in cols], fontsize=9, color=INK_2)
    ax.invert_yaxis()
    _style(ax, "")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("share of sentences (blind LLM tags)", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    fig.suptitle("Rhetorical devices per speech", x=0.02, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def anatomy_compare(ids: list[str], path: str, before_s: float = 6.0, after_s: float = 2.0) -> None:
    """Average loudness and pitch before applauded sentence endings, one line per speech."""
    from speechart.art.charts import smoothed_frames

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, dpi=150, facecolor=SURFACE)
    handles, labels = [], []
    for i, sid in enumerate(ids):
        d = load(sid)
        f = smoothed_frames(d["frames"])
        t = f.time.to_numpy()
        hop = t[1] - t[0]
        rel = np.arange(-before_s, after_s + 1e-9, hop)
        w = d["words_timeline"]
        ends = w.groupby("seg_id").tail(1)
        ends = ends[ends.reaction_after == "applause"]
        if ends.empty:
            continue
        rows = []
        for e in ends.end:
            idx = np.round((e + rel - t[0]) / hop).astype(int)
            ok = (idx >= 0) & (idx < len(t))
            rl, rp = np.full(len(rel), np.nan), np.full(len(rel), np.nan)
            rl[ok], rp[ok] = f.loud_s.to_numpy()[idx[ok]], f.pitch_s.to_numpy()[idx[ok]]
            # loudness relative to this speech's typical speaking level, pitch in semitones from its median
            rows.append((rl - np.nanmedian(f.loud_s), 12 * np.log2(rp / np.nanmedian(f.pitch_s))))
        for j, ax in enumerate(axes):
            arr = np.array([r[j] for r in rows])
            with np.errstate(all="ignore"):
                mean = np.where(np.sum(~np.isnan(arr), 0) >= 2, np.nanmean(arr, 0), np.nan)
            (h,) = ax.plot(rel, mean, color=SERIES[i], lw=2.2)
        handles.append(h)
        labels.append(f"{short_label(d['meta'])} (n={len(rows)})")
    for ax in axes:
        ax.axvline(0, color=INK_2, lw=0.8, ls=(0, (2, 2)))
        ax.axhline(0, color=GRID, lw=1)
    _style(axes[0], "loudness vs. speaker's\ntypical level (dB)")
    _style(axes[1], "pitch vs. speaker's\nmedian (semitones)")
    axes[1].set_xlabel("seconds from the end of an applauded sentence", fontsize=9, color=INK_2)
    axes[0].legend(handles, labels, frameon=False, ncol=2, fontsize=8, labelcolor=INK_2,
                   loc="upper left", bbox_to_anchor=(0, 1.22))
    fig.suptitle("The run-up to applause, compared", x=0.02, ha="left", fontsize=13, color=INK)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.83, bottom=0.09, hspace=0.12)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def echo_compare(ids: list[str], path: str, article: str = "Greta_Thunberg", days: int = 45) -> None:
    """Attention to the same Wikipedia article after each speech, aligned on the speech day."""
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150, facecolor=SURFACE)
    drew = False
    for i, sid in enumerate(ids):
        d = load(sid)
        p = d["out"] / "aftermath_wikipedia.csv"
        if not p.exists():
            continue
        v = pd.read_csv(p, parse_dates=["date"])
        v = v[v.article == article].set_index("date").views
        day0 = pd.Timestamp(d["meta"]["date"])
        rel = (v.index - day0).days
        keep = (rel >= -14) & (rel <= days)
        ax.plot(rel[keep], v[keep], color=SERIES[i], lw=2, label=short_label(d["meta"]))
        drew = True
    if not drew:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.axvline(0, color=INK_2, lw=0.8, ls=(0, (2, 2)))
    _style(ax, f"daily views of “{article.replace('_', ' ')}”\n(log scale)")
    ax.set_xlabel("days from the speech", fontsize=9, color=INK_2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    fig.suptitle("The echo of each speech on Wikipedia", x=0.02, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def contact_sheet(ids: list[str], image: str, path: str, labels: list[str], cols: int = 2,
                  bg: str = "#0e0d12", fg: str = "#e9e6f0") -> None:
    """Lay the same artwork from several speeches side by side."""
    from PIL import ImageDraw

    imgs = [Image.open(ROOT / "output" / i / "art" / image).convert("RGB") for i in ids]
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    rows = int(np.ceil(len(imgs) / cols))
    pad = 30
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * (h + 40) + (rows + 1) * pad), bg)
    draw = ImageDraw.Draw(sheet)
    for k, (im, lab) in enumerate(zip(imgs, labels)):
        r, c = divmod(k, cols)
        x, y = pad + c * (w + pad), pad + r * (h + 40 + pad)
        draw.text((x, y), lab, fill=fg)
        sheet.paste(im, (x, y + 40))
    sheet.thumbnail((3600, 3600))
    sheet.save(path)


def combined_tables(ids: list[str], out: Path) -> dict[str, int]:
    """Stack each speech's per-item tables into one long table per kind, with a speech column.

    These are the rows every number in metrics.csv and every comparison chart
    is computed from, so any cross-speech claim can be checked item by item.
    """
    counts = {}
    for name in ["sentences", "reaction_moments", "runup_curves"]:
        parts = []
        for sid in ids:
            p = ROOT / "output" / sid / f"{name}.csv"
            if p.exists():
                d = pd.read_csv(p)
                d.insert(0, "speech", short_label(load(sid)["meta"]))
                d.insert(0, "speech_id", sid)
                parts.append(d)
        if parts:
            table = pd.concat(parts, ignore_index=True)
            table.to_csv(out / f"{name}_all.csv", index=False)
            counts[name] = len(table)
    return counts
