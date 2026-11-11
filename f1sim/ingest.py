"""Download qualifying results with FastF1 (F1 live timing + Jolpica API).

Resumable by design:
- each qualifying session is saved as its own parquet file
- sessions already on disk are skipped on the next run
- FastF1 enforces hard API limits (Jolpica: 200 calls/hour, all APIs:
  500 calls/hour). When a limit is hit, the run stops cleanly and you rerun
  it after about an hour to continue where it left off.

CLI:
    python -m f1sim.ingest --start 2018 --end 2025 --data-dir /path/to/data
"""
import argparse
from dataclasses import dataclass, field

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from f1sim.paths import Paths, get_paths

Q_COLS = ["Q1", "Q2", "Q3"]
RESULT_COLS = [
    "DriverNumber",
    "Abbreviation",
    "DriverId",
    "FullName",
    "TeamName",
    "TeamId",
    "Position",
    *Q_COLS,
]


@dataclass
class IngestSummary:
    saved: list[str] = field(default_factory=list)
    already_on_disk: int = 0
    skipped_future: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    stopped_on_rate_limit: bool = False

    def report(self) -> str:
        lines = [
            f"Saved this run:     {len(self.saved)}",
            f"Already on disk:    {self.already_on_disk}",
            f"Skipped (future):   {len(self.skipped_future)}",
            f"Failed:             {len(self.failed)}",
        ]
        for label, err in self.failed:
            lines.append(f"  - {label}: {err}")
        if self.stopped_on_rate_limit:
            lines.append(
                "STOPPED: API rate limit reached. Wait ~60 minutes, then rerun - "
                "finished sessions are kept and skipped."
            )
        return "\n".join(lines)


def session_file(paths: Paths, year: int, rnd: int):
    return paths.quali_dir / f"{year}_{rnd:02d}.parquet"


def _load_quali_results(year: int, rnd: int) -> tuple[pd.DataFrame, str]:
    """Return (results, source).

    First try the cheap path: results only (Q times come from the Jolpica API).
    If Q times are missing, reload with lap data + race control messages so
    FastF1 can compute Q1/Q2/Q3 from the laps itself (deleted laps excluded).
    """
    session = fastf1.get_session(year, rnd, "Q")
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    res = session.results
    if res is not None and not res.empty and res[Q_COLS].notna().any().any():
        return res, "results_api"

    session = fastf1.get_session(year, rnd, "Q")
    session.load(laps=True, telemetry=False, weather=False, messages=True)
    return session.results, "computed_from_laps"


def ingest_season(year: int, paths: Paths, summary: IngestSummary) -> None:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    now = pd.Timestamp.now()

    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        label = f"{year} R{rnd:02d} {event['EventName']}"
        out = session_file(paths, year, rnd)

        if out.exists():
            summary.already_on_disk += 1
            continue
        if pd.notna(event["EventDate"]) and pd.Timestamp(event["EventDate"]) > now:
            summary.skipped_future.append(label)
            continue

        try:
            res, source = _load_quali_results(year, rnd)
        except RateLimitExceededError:
            raise
        except Exception as exc:  # missing session, API error, etc.
            summary.failed.append((label, f"{type(exc).__name__}: {exc}"))
            continue

        if res is None or res.empty or res[Q_COLS].isna().all().all():
            summary.failed.append((label, "no qualifying times available"))
            continue

        df = res[RESULT_COLS].copy().reset_index(drop=True)
        for col in Q_COLS:  # store lap times as seconds (float)
            df[col] = pd.to_timedelta(df[col]).dt.total_seconds()
        df.insert(0, "Year", year)
        df.insert(1, "Round", rnd)
        df.insert(2, "Event", event["EventName"])
        df["QTimeSource"] = source
        df.to_parquet(out, index=False)
        summary.saved.append(label)
        print(f"saved  {label}  ({source})")


def ingest_seasons(years, paths: Paths | None = None) -> IngestSummary:
    paths = paths or get_paths()
    fastf1.Cache.enable_cache(str(paths.cache_dir))
    summary = IngestSummary()
    try:
        for year in years:
            print(f"--- {year} ---")
            ingest_season(int(year), paths, summary)
    except RateLimitExceededError as exc:
        print(f"Rate limit: {exc}")
        summary.stopped_on_rate_limit = True
    print(summary.report())
    return summary


def load_quali(paths: Paths | None = None) -> pd.DataFrame:
    """All downloaded qualifying sessions as one DataFrame."""
    paths = paths or get_paths()
    files = sorted(paths.quali_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No qualifying data in {paths.quali_dir}. Run the ingest step first."
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    fastf1.set_log_level("WARNING")
    ingest_seasons(range(args.start, args.end + 1), get_paths(args.data_dir))


if __name__ == "__main__":
    main()
