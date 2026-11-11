"""Teammate qualifying gaps: first evidence for driver skill vs car performance.

For each race and team, the two teammates are compared in the LAST qualifying
segment they both set a time in (Q3 if both reached it, else Q2, else Q1):
same session, same track conditions, (nearly) the same car.
"""
import pandas as pd

SEGMENTS = ["Q3", "Q2", "Q1"]

# Gaps above this are usually a crash, a mistake or a technical problem rather
# than pace. They are flagged (not deleted) and left out of season summaries.
# This is a tunable assumption - check the flagged races before trusting it.
DEFAULT_OUTLIER_PCT = 2.0


def teammate_gaps(quali: pd.DataFrame, outlier_pct: float = DEFAULT_OUTLIER_PCT) -> pd.DataFrame:
    """One row per driver per race: gap to teammate in % (negative = faster)."""
    rows = []
    for (year, rnd, event, team), g in quali.groupby(["Year", "Round", "Event", "TeamName"]):
        if len(g) != 2:  # e.g. a substitute driver mid-weekend
            continue
        a, b = g.iloc[0], g.iloc[1]

        segment = next((s for s in SEGMENTS if pd.notna(a[s]) and pd.notna(b[s])), None)
        if segment is None:
            continue

        for me, mate in ((a, b), (b, a)):
            gap_pct = 100 * (me[segment] - mate[segment]) / mate[segment]
            rows.append(
                {
                    "Year": year,
                    "Round": rnd,
                    "Event": event,
                    "Team": team,
                    "Driver": me["Abbreviation"],
                    "DriverName": me["FullName"],
                    "Teammate": mate["Abbreviation"],
                    "Segment": segment,
                    "LapTime": me[segment],
                    "GapPct": gap_pct,
                    "Outlier": abs(gap_pct) > outlier_pct,
                }
            )
    return pd.DataFrame(rows).sort_values(["Year", "Round", "Team", "Driver"]).reset_index(drop=True)


def season_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    """Per season and teammate pairing: races compared, median gap, quali head-to-head."""
    clean = gaps[~gaps["Outlier"]]
    summary = (
        clean.groupby(["Year", "Team", "Driver", "Teammate"])
        .agg(
            Races=("GapPct", "size"),
            MedianGapPct=("GapPct", "median"),
            QualiWins=("GapPct", lambda s: int((s < 0).sum())),
        )
        .reset_index()
    )
    summary["H2H"] = (
        summary["QualiWins"].astype(str) + "-" + (summary["Races"] - summary["QualiWins"]).astype(str)
    )
    return summary.sort_values(["Year", "MedianGapPct"]).round({"MedianGapPct": 3}).reset_index(drop=True)
