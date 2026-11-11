"""Download raw race data with FastF1: every lap, session weather and results.

Saved per race to <data>/raw/race/:
    <year>_<round>_results.parquet  finishing order, grid, status, points
    <year>_<round>_weather.parquet  ~1 reading per minute (temps, rain, wind)
    <year>_<round>_laps.parquet     every lap of every driver (written LAST)

Nothing is cleaned or filtered here: which laps count as "clean" is decided
later, in the analysis, so that decision can change without re-downloading.
All lap and sector times are stored in seconds.

Resumable like the qualifying ingest: a race counts as done once its laps file
exists, and the run stops cleanly at FastF1's API rate limits (rerun after ~1h).

CLI:
    python -m f1sim.ingest_race --start 2018 --end 2025 --data-dir /path/to/data
"""
import argparse
import os
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from f1sim.ingest import IngestSummary, quiet_fastf1_logs
from f1sim.paths import Paths, get_paths

RESULT_COLS = [
    "DriverNumber", "Abbreviation", "DriverId", "FullName", "TeamName", "TeamId",
    "Position", "ClassifiedPosition", "GridPosition", "Status", "Points", "Laps", "Time",
]


def race_file(paths: Paths, year: int, rnd: int, kind: str) -> Path:
    return paths.race_dir / f"{year}_{rnd:02d}_{kind}.parquet"


def _timedeltas_to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="timedelta64[ns]").columns:
        df[col] = df[col].dt.total_seconds()
    return df


def _write(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file so an interrupted run never leaves a half file."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _tag(df: pd.DataFrame, year: int, rnd: int, event: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.insert(0, "Year", year)
    df.insert(1, "Round", rnd)
    df.insert(2, "Event", event)
    return df


def _ingest_race(year: int, rnd: int, event: str, paths: Paths, summary: IngestSummary) -> None:
    label = f"{year} R{rnd:02d} {event}"
    session = fastf1.get_session(year, rnd, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=True)

    try:
        laps = session.laps
    except Exception as exc:  # DataNotLoadedError when timing data is missing
        summary.failed.append((label, f"no lap data ({type(exc).__name__})"))
        return
    if laps is None or laps.empty:
        summary.failed.append((label, "no lap data"))
        return

    notes = []

    results = session.results
    if results is None or results.empty:
        notes.append("no results")
    else:
        cols = [c for c in RESULT_COLS if c in results.columns]
        _write(_tag(_timedeltas_to_seconds(results[cols]), year, rnd, event),
               race_file(paths, year, rnd, "results"))

    try:
        weather = session.weather_data
    except Exception:
        weather = None
    if weather is None or weather.empty:
        notes.append("no weather data")
    else:
        _write(_tag(_timedeltas_to_seconds(weather), year, rnd, event),
               race_file(paths, year, rnd, "weather"))

    laps_df = _timedeltas_to_seconds(pd.DataFrame(laps))
    if "Deleted" in laps_df:  # bool with gaps -> nullable boolean for parquet
        laps_df["Deleted"] = laps_df["Deleted"].astype("boolean")
    _write(_tag(laps_df, year, rnd, event), race_file(paths, year, rnd, "laps"))

    summary.saved.append(label)
    if notes:
        summary.warnings.append((label, ", ".join(notes)))
    print(f"saved  {label}  ({len(laps_df)} laps{'; ' + ', '.join(notes) if notes else ''})")


def ingest_race_season(year: int, paths: Paths, summary: IngestSummary) -> None:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    now = pd.Timestamp.now()

    for _, event in schedule.iterrows():
        rnd = int(event["RoundNumber"])
        name = event["EventName"]
        label = f"{year} R{rnd:02d} {name}"

        if race_file(paths, year, rnd, "laps").exists():
            summary.already_on_disk += 1
            continue
        if pd.notna(event["EventDate"]) and pd.Timestamp(event["EventDate"]) > now:
            summary.skipped_future.append(label)
            continue

        try:
            _ingest_race(year, rnd, name, paths, summary)
        except RateLimitExceededError:
            raise
        except Exception as exc:
            summary.failed.append((label, f"{type(exc).__name__}: {exc}"))


def ingest_race_seasons(years, paths: Paths | None = None) -> IngestSummary:
    paths = paths or get_paths()
    fastf1.Cache.configure(cache_dir=str(paths.cache_dir))
    summary = IngestSummary()
    try:
        for year in years:
            print(f"--- {year} ---")
            ingest_race_season(int(year), paths, summary)
    except RateLimitExceededError as exc:
        print(f"Rate limit: {exc}")
        summary.stopped_on_rate_limit = True
    print(summary.report())
    return summary


def load_race(kind: str, paths: Paths | None = None, years=None) -> pd.DataFrame:
    """All downloaded races for one kind: 'laps', 'weather' or 'results'."""
    paths = paths or get_paths()
    files = sorted(paths.race_dir.glob(f"*_{kind}.parquet"))
    if years is not None:
        wanted = {int(y) for y in years}
        files = [f for f in files if int(f.name[:4]) in wanted]
    if not files:
        raise FileNotFoundError(f"No race {kind} data in {paths.race_dir}. Run the race ingest first.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    quiet_fastf1_logs(True)
    ingest_race_seasons(range(args.start, args.end + 1), get_paths(args.data_dir))


if __name__ == "__main__":
    main()
