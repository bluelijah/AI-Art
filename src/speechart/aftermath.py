"""Aftermath: how the speech echoed in the world afterwards.

Daily time series around the speech date, on their own (day-scale) axis:

- Wikipedia pageviews (Wikimedia REST API, free, goes back to 2015):
  attention to the speaker, the event and related movements.
- GDELT DOC 2.0 news volume for a phrase (free, but rate-limited; it often
  answers 429 Too Many Requests, so failures are reported, not fatal).

We frame this as resonance, not impact: a spike after the speech is an
echo, not proof the speech caused it.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.parse
import urllib.request
from datetime import date

import pandas as pd

USER_AGENT = "speechart-research/0.1 (art and research project)"


def _get(url: str, retries: int = 3, backoff_s: float = 10.0) -> str:
    import certifi

    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError("unreachable")


def wikipedia_pageviews(articles: list[str], start: date, end: date, project: str = "en.wikipedia") -> pd.DataFrame:
    """Daily human pageviews: columns date, article, views."""
    rows = []
    for art in articles:
        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/user/"
               f"{urllib.parse.quote(art, safe='')}/daily/{start:%Y%m%d}/{end:%Y%m%d}")
        for item in json.loads(_get(url))["items"]:
            rows.append((pd.to_datetime(item["timestamp"][:8]), art, item["views"]))
    return pd.DataFrame(rows, columns=["date", "article", "views"])


def gdelt_volume(query: str, start: date, end: date) -> pd.DataFrame:
    """Daily count of news articles matching `query`: columns date, query, articles."""
    url = ("https://api.gdeltproject.org/api/v2/doc/doc?mode=timelinevolraw&format=json"
           f"&query={urllib.parse.quote(query)}&startdatetime={start:%Y%m%d}000000&enddatetime={end:%Y%m%d}235959")
    data = json.loads(_get(url))
    series = data["timeline"][0]["data"]
    df = pd.DataFrame([(pd.to_datetime(p["date"][:8]), query, p["value"]) for p in series],
                      columns=["date", "query", "articles"])
    return df.groupby(["date", "query"], as_index=False)["articles"].sum()


def echo_summary(views: pd.DataFrame, speech_day: date, baseline_days: int = 14) -> pd.DataFrame:
    """Per article: baseline (median of the days before), peak, peak day, and peak / baseline."""
    d = pd.Timestamp(speech_day)
    out = []
    for art, g in views.groupby("article"):
        before = g[(g.date < d) & (g.date >= d - pd.Timedelta(days=baseline_days))].views
        after = g[g.date >= d]
        peak = after.loc[after.views.idxmax()]
        base = before.median() if len(before) else float("nan")
        out.append((art, base, peak.views, peak.date.date(), (peak.date - d).days, peak.views / base))
    return pd.DataFrame(out, columns=["article", "baseline", "peak", "peak_date", "days_after", "x_baseline"])
