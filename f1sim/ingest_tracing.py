"""Race laps, weather and race control messages from TracingInsights (free).

Why this exists: since October 2026 F1's timing APIs need an F1 TV login, so
FastF1 can no longer download lap data without a subscription.
TracingInsights published the same FastF1-derived data on
GitHub before that, one public repo per season, licensed Apache-2.0:
    https://github.com/TracingInsights/<year>

For each race we use three files from "<Grand Prix>/Race/":
    session_laptimes.json  every lap of every driver (42 fields, see below)
    weather.json           ~1 reading per minute
    rcm.json               race control messages (flags, SC/VSC, penalties)

Only those files are downloaded (blobless clone + sparse checkout); the repos'
telemetry, radio and cache folders are skipped. The git commit used is stored
in every row (SourceCommit) so results can be traced to an exact data version.

Values are kept raw apart from two fixes: the text "None" becomes a real
missing value, and columns are given proper types. Column names stay as in the
source; key lap fields:
    drv/dNum/team   driver code, car number, team
    lap, time       lap number, lap time (s)       s1/s2/s3  sector times (s)
    compound, life, fresh, stint                   tyre compound, age (laps), new?, stint
    status          track status codes for the lap, e.g. "1" green, "4" safety
                    car, "6" VSC, "7" VSC ending; "126" = several during the lap
    pin, pout       pit in / pit out session time (s)
    pos             position at end of lap
    iacc, del, delR, ff1G   FastF1 accuracy flag, deleted lap + reason, generated lap
    wT ... wWS      weather matched to the lap (air/track temp, rain, wind ...)
"""
import json
import os
import subprocess
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from f1sim.ingest import IngestSummary
from f1sim.paths import Paths, get_paths

REPO_URL = "https://github.com/TracingInsights/{year}.git"
DEFAULT_CLONE_DIR = Path("/content/tracinginsights") if Path("/content").exists() else Path("tracinginsights_repos")

RACE_FILES = {
    "laps": "session_laptimes.json",
    "weather": "weather.json",
    "rcm": "rcm.json",
}
# Codes and labels that must stay text even though they look numeric
# ("126" is three track-status flags, not the number 126).
KEEP_AS_TEXT = {"status", "dNum", "drv", "team", "compound", "delR", "lSD",
                "msg", "cat", "flag", "scope"}


def _git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],  # keep e.g. "São Paulo" readable
        cwd=cwd, check=True, capture_output=True, text=True,
    )
    return out.stdout


def fetch_season_repo(year: int, clone_dir: Path = DEFAULT_CLONE_DIR) -> tuple[Path, str]:
    """Download only the race-level files for one season. Returns (repo path, commit)."""
    repo = Path(clone_dir) / str(year)
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--quiet", "--filter=blob:none", "--no-checkout", "--depth", "1",
             REPO_URL.format(year=year), str(repo))
        _git("sparse-checkout", "set", "--no-cone",
             *[f"/*/Race/{name}" for name in RACE_FILES.values()], cwd=repo)
        _git("checkout", "--quiet", cwd=repo)
    return repo, _git("rev-parse", "HEAD", cwd=repo).strip()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns[df.dtypes == object]:
        df[col] = df[col].where(df[col].map(lambda v: v != "None"))
    for col in df.columns:
        if col in KEEP_AS_TEXT or df[col].dtype != object:
            continue
        values = df[col].dropna()
        if len(values) and values.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            df[col] = df[col].astype("boolean")
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == df[col].notna().sum():  # only if nothing is lost
            df[col] = numeric
    for col in KEEP_AS_TEXT & set(df.columns):
        df[col] = df[col].astype("string")
    return df


def read_json_table(path: Path) -> pd.DataFrame:
    """Read a TracingInsights column-oriented JSON file ({field: [values...]})."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
        raise ValueError(f"{path.name}: expected {{field: [values]}}")
    lengths = {len(v) for v in data.values()}
    if len(lengths) != 1:
        raise ValueError(f"{path.name}: fields have different lengths {sorted(lengths)}")
    return _clean(pd.DataFrame(data))


def race_ti_file(paths: Paths, year: int, rnd: int, kind: str) -> Path:
    return paths.race_ti_dir / f"{year}_{rnd:02d}_{kind}.parquet"


def _write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def ingest_tracing_season(year: int, paths: Paths, summary: IngestSummary,
                          clone_dir: Path = DEFAULT_CLONE_DIR, force: bool = False) -> None:
    repo, commit = fetch_season_repo(year, clone_dir)
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rounds = {name: int(rnd) for name, rnd in zip(schedule["EventName"], schedule["RoundNumber"])}

    race_dirs = sorted(p.parent for p in repo.glob("*/Race/session_laptimes.json"))
    matched = set()
    for race_dir in race_dirs:
        event = race_dir.parent.name
        if event not in rounds:
            summary.failed.append((f"{year} {event}", "folder name not in FastF1 schedule"))
            continue
        rnd = rounds[event]
        matched.add(event)
        label = f"{year} R{rnd:02d} {event}"

        if race_ti_file(paths, year, rnd, "laps").exists() and not force:
            summary.already_on_disk += 1
            continue

        notes = []
        try:
            for kind in ("weather", "rcm", "laps"):  # laps last: marks the race done
                src = race_dir / RACE_FILES[kind]
                if not src.exists():
                    notes.append(f"no {kind} file")
                    continue
                df = read_json_table(src)
                df.insert(0, "Year", year)
                df.insert(1, "Round", rnd)
                df.insert(2, "Event", event)
                df["SourceCommit"] = commit
                _write(df, race_ti_file(paths, year, rnd, kind))
        except Exception as exc:
            summary.failed.append((label, f"{type(exc).__name__}: {exc}"))
            continue

        summary.saved.append(label)
        if notes:
            summary.warnings.append((label, ", ".join(notes)))

    for event, rnd in sorted(rounds.items(), key=lambda kv: kv[1]):
        if event not in matched:
            summary.failed.append((f"{year} R{rnd:02d} {event}", "no race folder in TracingInsights repo"))


def ingest_tracing_seasons(years, paths: Paths | None = None,
                           clone_dir: Path = DEFAULT_CLONE_DIR, force: bool = False) -> IngestSummary:
    paths = paths or get_paths()
    fastf1.Cache.configure(cache_dir=str(paths.cache_dir))
    summary = IngestSummary()
    for year in years:
        print(f"--- {year} ---")
        before = len(summary.saved)
        ingest_tracing_season(int(year), paths, summary, clone_dir, force)
        print(f"saved {len(summary.saved) - before} races")
    print(summary.report())
    return summary


def load_race_ti(kind: str, paths: Paths | None = None, years=None) -> pd.DataFrame:
    """All races for one kind: 'laps', 'weather' or 'rcm'."""
    paths = paths or get_paths()
    files = sorted(paths.race_ti_dir.glob(f"*_{kind}.parquet"))
    if years is not None:
        wanted = {int(y) for y in years}
        files = [f for f in files if int(f.name[:4]) in wanted]
    if not files:
        raise FileNotFoundError(f"No {kind} data in {paths.race_ti_dir}. Run the TracingInsights ingest first.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
