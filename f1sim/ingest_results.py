"""Race results (finishing order, grid, status, points) from the Jolpica API.

Jolpica (the successor to Ergast) is free and needs no F1 TV login. Its
results endpoint returns up to 100 result rows per request, so a whole season
is ~5 requests and 2018-2025 finishes in one run.

Saved per season to <data>/raw/race_results/<year>.parquet, one row per
driver per race. Values are kept as Jolpica gives them, notably:
    Position            finishing order, including non-finishers (1..N)
    ClassifiedPosition  "1".."20" if classified, else "R" retired,
                        "D" disqualified, "W" withdrawn, "N" not classified...
    GridPosition        starting grid slot; 0 means started from the pit lane
    Status              "Finished", "+1 Lap", or the retirement reason
                        ("Engine", "Collision", ...)
    TimeMillis          winner's race time / finishers' total time in ms
                        (empty for lapped cars and non-finishers)

Resumable: seasons already on disk are skipped (force=True re-downloads).
"""
import time

import pandas as pd
import requests

from f1sim.ingest import IngestSummary
from f1sim.paths import Paths, get_paths

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_SIZE = 100
PAUSE_S = 0.5  # stay well under Jolpica's per-second limit


class JolpicaRateLimited(Exception):
    pass


def _get(session: requests.Session, url: str, params: dict) -> dict:
    resp = session.get(url, params=params, timeout=30)
    if resp.status_code == 429:
        raise JolpicaRateLimited("Jolpica rate limit hit (HTTP 429). Wait ~1 hour and rerun.")
    resp.raise_for_status()
    time.sleep(PAUSE_S)
    return resp.json()


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row(year: int, race: dict, res: dict) -> dict:
    driver, team = res.get("Driver", {}), res.get("Constructor", {})
    return {
        "Year": year,
        "Round": _int(race.get("round")),
        "Event": race.get("raceName"),
        "Date": race.get("date"),
        "DriverNumber": res.get("number"),
        "Abbreviation": driver.get("code"),
        "DriverId": driver.get("driverId"),
        "FullName": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
        "TeamName": team.get("name"),
        "TeamId": team.get("constructorId"),
        "Position": _int(res.get("position")),
        "ClassifiedPosition": res.get("positionText"),
        "GridPosition": _int(res.get("grid")),
        "Points": _float(res.get("points")),
        "Laps": _int(res.get("laps")),
        "Status": res.get("status"),
        "TimeMillis": _float(res.get("Time", {}).get("millis")),
    }


def fetch_season_results(year: int, session: requests.Session | None = None) -> pd.DataFrame:
    """All race results for one season, following Jolpica's pagination."""
    session = session or requests.Session()
    url = f"{BASE_URL}/{year}/results.json"
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        data = _get(session, url, {"limit": PAGE_SIZE, "offset": offset})
        mr = data["MRData"]
        total = int(mr["total"])
        for race in mr["RaceTable"]["Races"]:  # a race can span two pages
            rows.extend(_row(year, race, res) for res in race.get("Results", []))
        offset += PAGE_SIZE
    df = pd.DataFrame(rows)
    if len(df) != total:
        raise ValueError(f"{year}: got {len(df)} result rows but Jolpica reports {total}")
    return df


def check_season(df: pd.DataFrame) -> list[str]:
    """Data checks; returns a list of problems (empty if all fine)."""
    problems = []
    for (rnd, event), g in df.groupby(["Round", "Event"]):
        n = len(g)
        if g["DriverId"].duplicated().any():
            problems.append(f"R{rnd:02d} {event}: a driver appears twice")
        if sorted(g["Position"].dropna().astype(int)) != list(range(1, n + 1)):
            problems.append(f"R{rnd:02d} {event}: positions aren't 1..{n}")
        if n < 18:
            problems.append(f"R{rnd:02d} {event}: only {n} drivers")
    return problems


def results_file(paths: Paths, year: int):
    return paths.results_dir / f"{year}.parquet"


def ingest_results_seasons(years, paths: Paths | None = None, force: bool = False) -> IngestSummary:
    paths = paths or get_paths()
    summary = IngestSummary()
    session = requests.Session()
    try:
        for year in years:
            year = int(year)
            out = results_file(paths, year)
            if out.exists() and not force:
                summary.already_on_disk += 1
                continue
            try:
                df = fetch_season_results(year, session)
            except JolpicaRateLimited:
                raise
            except Exception as exc:
                summary.failed.append((str(year), f"{type(exc).__name__}: {exc}"))
                continue
            if df.empty:
                summary.failed.append((str(year), "no results returned"))
                continue
            for problem in check_season(df):
                summary.warnings.append((str(year), problem))
            tmp = out.with_name(out.name + ".tmp")
            df.to_parquet(tmp, index=False)
            tmp.replace(out)
            summary.saved.append(str(year))
            print(f"saved  {year}: {df['Round'].nunique()} races, {len(df)} rows")
    except JolpicaRateLimited as exc:
        print(exc)
        summary.stopped_on_rate_limit = True
    print(summary.report())
    return summary


def load_results(paths: Paths | None = None, years=None) -> pd.DataFrame:
    paths = paths or get_paths()
    files = sorted(paths.results_dir.glob("*.parquet"))
    if years is not None:
        wanted = {int(y) for y in years}
        files = [f for f in files if int(f.stem) in wanted]
    if not files:
        raise FileNotFoundError(f"No race results in {paths.results_dir}. Run the results ingest first.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
