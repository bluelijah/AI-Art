"""Compare speeches side by side.

Usage:
    uv run python scripts/compare.py thunberg_2019_un thunberg_2018_cop24 malala_2013_un mlk_1963_dream

Each speech must have been through run_pipeline.py. Writes to output/compare/:
  metrics.csv              one row of comparable numbers per speech (the summary)
  sentences_all.csv        every sentence of every speech, with tags and delivery (the process)
  reaction_moments_all.csv every audience reaction with what preceded it
  runup_curves_all.csv     loudness / pitch around every sentence ending
  speech_profiles.png      small multiples: pace, pauses, range, applause, prediction
  rhetorical_devices.png   share of sentences using each device
  applause_runup.png       loudness and pitch before applauded sentence endings
  echo_wikipedia.png       attention to the speaker after each speech (where data exists)
  fingerprints.png, landscapes.png   the same artwork for each speech, side by side
"""

import sys
from pathlib import Path

from speechart import compare

ROOT = Path(__file__).resolve().parents[1]


def main(ids: list[str]) -> None:
    out = ROOT / "output" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    table = compare.metrics_table(ids)
    table.to_csv(out / "metrics.csv", index=False)
    print(table.drop(columns=[c for c in table if c.startswith("dev_")]).round(2).T.to_string(header=False))

    counts = compare.combined_tables(ids, out)
    print("\nper-item tables:", ", ".join(f"{k}_all.csv ({v} rows)" for k, v in counts.items()))

    compare.profile_chart(table, out / "speech_profiles.png")
    compare.device_chart(table, out / "rhetorical_devices.png")
    compare.anatomy_compare(ids, out / "applause_runup.png")
    compare.echo_compare(ids, out / "echo_wikipedia.png")
    labels = list(table.speech)
    compare.contact_sheet(ids, "speech_fingerprint.png", out / "fingerprints.png", labels)
    compare.contact_sheet(ids, "applause_landscape.png", out / "landscapes.png", labels, cols=1,
                          bg="#efe8d6", fg="#2d261c")
    print(f"\nwrote {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main(sys.argv[1:] or ["thunberg_2019_un"])
