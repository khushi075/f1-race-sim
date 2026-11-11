"""Manually entered qualifying times for sessions no API has times for.

Source file: manual_data/quali_times.csv (committed to git, so it's reviewable).
Rows are matched to FastF1's results by car number (DriverNumber). Driver and
team identity always comes from FastF1/Jolpica; only Q1/Q2/Q3 come from here.

Currently covers:
- 2025 R06 Miami: F1 live timing has no data and Jolpica has the order but no
  times. Times copied from the official formula1.com qualifying results.
"""
import re

import numpy as np
import pandas as pd

from f1sim.paths import MANUAL_QUALI_CSV

Q_COLS = ["Q1", "Q2", "Q3"]
_LAP_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}\.\d{1,3})$")


def parse_lap_time(value) -> float:
    """'1:26.204' -> 86.204 seconds; blank -> NaN. Raises on anything else."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    m = _LAP_RE.match(text)
    if not m:
        raise ValueError(f"Unrecognised lap time: {value!r}")
    minutes = int(m.group(1) or 0)
    return minutes * 60 + float(m.group(2))


def load_manual_times(year: int, rnd: int) -> pd.DataFrame | None:
    """Manual Q times (seconds) for one session, or None if there are none."""
    if not MANUAL_QUALI_CSV.exists():
        return None
    manual = pd.read_csv(MANUAL_QUALI_CSV, dtype={"DriverNumber": str}, keep_default_na=False)
    rows = manual[(manual["Year"] == year) & (manual["Round"] == rnd)].copy()
    if rows.empty:
        return None
    for col in Q_COLS:
        rows[col] = rows[col].map(parse_lap_time)
    return rows.set_index("DriverNumber")[Q_COLS]


def apply_manual_times(results: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame | None:
    """Fill Q1-Q3 in FastF1 results from the manual file.

    Returns None if there is no manual data for this session. Raises if the
    manual drivers don't match FastF1's driver list exactly, so a typo can't
    silently drop or misassign a driver.
    """
    manual = load_manual_times(year, rnd)
    if manual is None:
        return None

    api_drivers = set(results["DriverNumber"].astype(str))
    manual_drivers = set(manual.index)
    if api_drivers != manual_drivers:
        raise ValueError(
            f"Manual times for {year} R{rnd:02d} don't match FastF1's driver list. "
            f"Missing from CSV: {sorted(api_drivers - manual_drivers)}; "
            f"not in FastF1 results: {sorted(manual_drivers - api_drivers)}"
        )

    out = results.copy()
    keys = out["DriverNumber"].astype(str)
    for col in Q_COLS:
        out[col] = keys.map(manual[col]).to_numpy()
    return out
