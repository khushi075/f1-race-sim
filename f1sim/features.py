"""Teammate qualifying gaps: first evidence for driver skill vs car performance.

For each race and team, the two teammates are compared in the LAST qualifying
segment they both set a time in (Q3 if both reached it, else Q2, else Q1):
same session, same track conditions, (nearly) the same car.
"""
import pandas as pd

SEGMENTS = ["Q3", "Q2", "Q1"]

# Gaps above this are usually a crash, a mistake, a technical problem or
# changing track conditions rather than pace. Both teammates in the race are
# flagged (not deleted) and left out of season summaries.
# This is a tunable assumption - check the flagged races before trusting it.
DEFAULT_OUTLIER_PCT = 2.0


def _with_team_key(quali: pd.DataFrame) -> pd.DataFrame:
    """Add TeamKey (stable id) and Team (one display name per team per season).

    Team names vary by data source (e.g. "Red Bull Racing" from F1 timing vs
    "Red Bull" from Jolpica), so teams are grouped by Jolpica's TeamId instead
    and shown under their most common name that season.
    """
    q = quali.copy()
    q["TeamKey"] = q["TeamId"].replace("", pd.NA).fillna(q["TeamName"])
    display = (
        q.groupby(["Year", "TeamKey"])["TeamName"]
        .agg(lambda s: s.value_counts().index[0])
        .rename("Team")
    )
    return q.join(display, on=["Year", "TeamKey"])


def teammate_gaps(quali: pd.DataFrame, outlier_pct: float = DEFAULT_OUTLIER_PCT) -> pd.DataFrame:
    """One row per driver per race: gap to teammate in % (negative = faster)."""
    rows = []
    quali = _with_team_key(quali)
    for (year, rnd, event, _, team), g in quali.groupby(["Year", "Round", "Event", "TeamKey", "Team"]):
        if len(g) != 2:  # e.g. a substitute driver mid-weekend
            continue
        a, b = g.iloc[0], g.iloc[1]

        segment = next((s for s in SEGMENTS if pd.notna(a[s]) and pd.notna(b[s])), None)
        if segment is None:
            continue

        gap_a = 100 * (a[segment] - b[segment]) / b[segment]
        gap_b = 100 * (b[segment] - a[segment]) / a[segment]
        # Flag the pair as a whole: the two gaps differ slightly in size (each is
        # relative to the other driver's lap), so flagging per driver could
        # exclude a race for one teammate but not the other.
        pair_outlier = max(abs(gap_a), abs(gap_b)) > outlier_pct

        for me, mate, gap_pct in ((a, b, gap_a), (b, a, gap_b)):
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
                    "Outlier": pair_outlier,
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
