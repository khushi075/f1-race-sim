"""Teammate race-pace gaps from a small lap-time model per race.

Clean laps: only laps FastF1 marks as accurate (`iacc`). That flag already
requires no pit in/out lap, a real (not generated) lap time, sector times that
add up to the lap time, and track status green or yellow only (so safety car,
VSC and red-flag laps are out).

Wet races are excluded for now. Two definitions (see wet_races):
"tyres" (default: intermediate/wet tyres used) or "tyres_or_rain" (also any
rain recorded during a lap, which catches races that stayed dry enough for
slicks).

Model, fitted separately for each race on its clean dry laps:

    lap_time ~ driver + compound + compound:tyre_age + lap_number

- driver:              each driver's pace, the part we want
- compound:            soft/medium/hard etc. have different base pace
- compound:tyre_age    each compound degrades at its own rate per lap of age
- lap_number:          fuel burn-off and track rubbering-in (cars get faster)

By default it's a robust regression (Huber), so laps slowed by traffic or a
mistake pull less on the estimates than with ordinary least squares.

A teammate gap is the difference between the two drivers' coefficients: the
lap-time difference with tyres and fuel load held equal. It's reported in
seconds and as % of the race's median clean lap, with a standard error from
the model's covariance (so close gaps can be told apart from noise).
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
NOT_A_COMPOUND = {"UNKNOWN", "TEST_UNKNOWN"}
DEFAULT_MIN_LAPS = 10  # a driver needs at least this many clean laps in a race
FORMULA = "time ~ C(drv) + C(compound) + C(compound):life + lap"


WET_RULES = ("tyres", "tyres_or_rain")


def wet_races(laps: pd.DataFrame, rule: str = "tyres") -> pd.DataFrame:
    """Races counted as wet, with the evidence.

    rule="tyres":         any intermediate/wet tyre lap (default)
    rule="tyres_or_rain": also any rain recorded during a lap (stricter; catches
                          light drizzle even when everyone stayed on slicks)
    """
    if rule not in WET_RULES:
        raise ValueError(f"rule must be one of {WET_RULES}")
    g = laps.groupby(["Year", "Round", "Event"])
    wet = pd.DataFrame({
        "WetTyreLaps": g["compound"].agg(lambda s: int(s.isin(WET_COMPOUNDS).sum())),
        "RainLaps": g["wR"].agg(lambda s: int(s.fillna(False).astype(bool).sum())),
    }).reset_index()
    is_wet = wet["WetTyreLaps"] > 0
    if rule == "tyres_or_rain":
        is_wet |= wet["RainLaps"] > 0
    return wet[is_wet].reset_index(drop=True)


def dry_races(laps: pd.DataFrame, wet_rule: str = "tyres") -> pd.DataFrame:
    """All races not counted as wet: the races the model should cover."""
    races = laps[["Year", "Round", "Event"]].drop_duplicates()
    wet = wet_races(laps, wet_rule)[["Year", "Round"]].assign(_wet=True)
    races = races.merge(wet, on=["Year", "Round"], how="left")
    return races[races["_wet"].isna()].drop(columns="_wet").reset_index(drop=True)


def clean_dry_laps(laps: pd.DataFrame, exclude_wet: bool = True,
                   wet_rule: str = "tyres") -> pd.DataFrame:
    """Accurate laps on known dry compounds, optionally without wet races.

    Laps without a compound or tyre age are dropped too: the model needs both.
    """
    keep = (
        laps["iacc"].fillna(False).astype(bool)
        & laps["time"].notna()
        & laps["life"].notna()
        & laps["compound"].notna()
        & ~laps["compound"].isin(WET_COMPOUNDS | NOT_A_COMPOUND)
    )
    out = laps[keep]
    if exclude_wet:
        wet = wet_races(laps, wet_rule)[["Year", "Round"]].assign(_wet=True)
        out = out.merge(wet, on=["Year", "Round"], how="left")
        out = out[out["_wet"].isna()].drop(columns="_wet")
    return out.reset_index(drop=True)


def _fit(race: pd.DataFrame, robust: bool):
    if robust:
        return smf.rlm(FORMULA, race, M=sm.robust.norms.HuberT()).fit()
    return smf.ols(FORMULA, race).fit()


def _driver_term(params: pd.Index, drv: str) -> str | None:
    name = f"C(drv)[T.{drv}]"
    return name if name in params else None  # None = the baseline driver (effect 0)


class NotIdentifiable(Exception):
    """The race's data can't separate every driver's pace from tyre effects."""


def _prepare(race: pd.DataFrame, min_laps: int) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Apply the per-race data rules. Returns (laps to fit, [(driver, note)])."""
    notes = []
    race = race.copy()
    race["drv"] = race["drv"].astype(str)
    race["compound"] = race["compound"].astype(str)

    counts = race["drv"].value_counts()
    race = race[race["drv"].isin(counts[counts >= min_laps].index)]

    # A compound used by only one driver can't be told apart from that
    # driver's own pace (e.g. 2018 Monza: Magnussen alone on mediums), so
    # those laps are dropped.
    users = race.groupby("compound")["drv"].nunique()
    for comp in users[users < 2].index:
        solo = race[race["compound"] == comp]
        notes.append((solo["drv"].iloc[0],
                      f"only driver on {comp}: {len(solo)} laps dropped"))
        race = race[race["compound"] != comp]

    counts = race["drv"].value_counts()
    for drv in counts[counts < min_laps].index:
        if any(d == drv for d, _ in notes):
            notes.append((drv, f"under {min_laps} clean laps after that; driver excluded"))
    race = race[race["drv"].isin(counts[counts >= min_laps].index)]
    return race, notes


def _check_identifiable(race: pd.DataFrame) -> None:
    X = patsy.dmatrix(FORMULA.split("~", 1)[1], race, return_type="dataframe")
    rank = np.linalg.matrix_rank(X.values)
    if rank < X.shape[1]:
        raise NotIdentifiable(
            f"model can't separate all effects (rank {rank} < {X.shape[1]} terms)")


def race_teammate_gaps(race: pd.DataFrame, min_laps: int = DEFAULT_MIN_LAPS,
                       robust: bool = True) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Teammate gaps for one race's clean dry laps.

    Returns (one row per driver, [(driver, note)] for drivers or laps dropped).
    Raises NotIdentifiable if the data can't separate driver and tyre effects.
    """
    race, notes = _prepare(race, min_laps)
    if race["drv"].nunique() < 2:
        return pd.DataFrame(), notes
    _check_identifiable(race)

    counts = race["drv"].value_counts()
    res = _fit(race, robust)
    params, cov = res.params, res.cov_params()
    ref_lap = race["time"].median()
    year, rnd, event = race[["Year", "Round", "Event"]].iloc[0]

    rows = []
    for team, g in race.groupby("team"):
        drivers = sorted(g["drv"].unique())
        if len(drivers) != 2:
            continue
        a, b = drivers
        ta, tb = _driver_term(params.index, a), _driver_term(params.index, b)
        # gap = effect(a) - effect(b); baseline driver's effect is 0
        vec = pd.Series(0.0, index=params.index)
        if ta: vec[ta] += 1
        if tb: vec[tb] -= 1
        gap = float(vec @ params)
        se = float(np.sqrt(vec @ cov @ vec))
        for me, mate, sign in ((a, b, 1), (b, a, -1)):
            rows.append({
                "Year": year, "Round": rnd, "Event": event, "Team": team,
                "Driver": me, "Teammate": mate,
                "GapSec": sign * gap, "GapSE": se,
                "GapPct": 100 * sign * gap / ref_lap,
                "GapSEPct": 100 * se / ref_lap,
                "Laps": int(counts[me]), "TeammateLaps": int(counts[mate]),
            })
    return pd.DataFrame(rows), notes


def teammate_race_gaps(clean: pd.DataFrame, min_laps: int = DEFAULT_MIN_LAPS,
                       robust: bool = True,
                       expected_races: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gaps for every race.

    Returns (gaps, skipped): skipped lists whole races (Driver empty) and
    individual drivers/laps the model had to leave out, each with the reason.
    Pass expected_races (e.g. dry_races(laps)) so races with no clean laps at
    all, such as 2021 Belgium (run entirely behind the safety car), are listed
    too instead of disappearing silently.
    """
    out, skipped = [], []
    if expected_races is not None:
        present = set(map(tuple, clean[["Year", "Round"]].drop_duplicates().values))
        for year, rnd, event in expected_races[["Year", "Round", "Event"]].itertuples(index=False):
            if (year, rnd) not in present:
                skipped.append((year, rnd, event, None, "no clean dry laps (e.g. whole race behind the safety car)"))
    for (year, rnd, event), race in clean.groupby(["Year", "Round", "Event"]):
        try:
            gaps, notes = race_teammate_gaps(race, min_laps, robust)
        except Exception as exc:
            skipped.append((year, rnd, event, None, f"{type(exc).__name__}: {exc}"))
            continue
        skipped.extend((year, rnd, event, drv, note) for drv, note in notes)
        if gaps.empty:
            skipped.append((year, rnd, event, None, "no teammate pair with enough clean laps"))
        else:
            out.append(gaps)
    gaps = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    skipped = (pd.DataFrame(skipped, columns=["Year", "Round", "Event", "Driver", "Reason"])
               .sort_values(["Year", "Round"]).reset_index(drop=True))
    return gaps.sort_values(["Year", "Round", "Team", "Driver"]).reset_index(drop=True), skipped


def race_season_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    """Per season and pairing: races, median gap, race-pace head-to-head."""
    s = (
        gaps.groupby(["Year", "Team", "Driver", "Teammate"])
        .agg(Races=("GapPct", "size"),
             MedianGapPct=("GapPct", "median"),
             MedianGapSec=("GapSec", "median"),
             PaceWins=("GapPct", lambda x: int((x < 0).sum())))
        .reset_index()
    )
    s["H2H"] = s["PaceWins"].astype(str) + "-" + (s["Races"] - s["PaceWins"]).astype(str)
    return s.sort_values(["Year", "MedianGapPct"]).round({"MedianGapPct": 3, "MedianGapSec": 3}).reset_index(drop=True)


def race_driver_pace(race: pd.DataFrame, min_laps: int = DEFAULT_MIN_LAPS,
                     robust: bool = True) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Every driver's race pace relative to the race's average driver.

    Same per-race model and data rules as race_teammate_gaps. Each driver's
    coefficient is centred on the mean of all drivers in the race, and its
    standard error comes from the full covariance of that contrast.
    PacePct > 0 means slower than the average driver in that race.
    """
    race, notes = _prepare(race, min_laps)
    if race["drv"].nunique() < 2:
        return pd.DataFrame(), notes
    _check_identifiable(race)

    counts = race["drv"].value_counts()
    res = _fit(race, robust)
    params, cov = res.params, res.cov_params()
    ref_lap = race["time"].median()
    year, rnd, event = race[["Year", "Round", "Event"]].iloc[0]

    drivers = sorted(race["drv"].unique())
    # A maps parameters -> each driver's effect (baseline driver = 0)
    A = pd.DataFrame(0.0, index=drivers, columns=params.index)
    for d in drivers:
        term = _driver_term(params.index, d)
        if term:
            A.loc[d, term] = 1.0
    centred = A - A.mean(axis=0)             # effect minus the race average
    effects = centred.values @ params.values
    covs = centred.values @ cov.values @ centred.values.T
    team = race.groupby("drv")["team"].first()

    rows = [{
        "Year": year, "Round": rnd, "Event": event, "Driver": d, "Team": team[d],
        "PaceSec": float(effects[i]), "PaceSESec": float(np.sqrt(covs[i, i])),
        "PacePct": float(100 * effects[i] / ref_lap),
        "PaceSEPct": float(100 * np.sqrt(covs[i, i]) / ref_lap),
        "Laps": int(counts[d]),
    } for i, d in enumerate(drivers)]
    return pd.DataFrame(rows), notes


def driver_race_pace(clean: pd.DataFrame, min_laps: int = DEFAULT_MIN_LAPS,
                     robust: bool = True,
                     expected_races: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """race_driver_pace for every race. Returns (pace, skipped) like teammate_race_gaps."""
    out, skipped = [], []
    if expected_races is not None:
        present = set(map(tuple, clean[["Year", "Round"]].drop_duplicates().values))
        for year, rnd, event in expected_races[["Year", "Round", "Event"]].itertuples(index=False):
            if (year, rnd) not in present:
                skipped.append((year, rnd, event, None, "no clean dry laps (e.g. whole race behind the safety car)"))
    for (year, rnd, event), race in clean.groupby(["Year", "Round", "Event"]):
        try:
            pace, notes = race_driver_pace(race, min_laps, robust)
        except Exception as exc:
            skipped.append((year, rnd, event, None, f"{type(exc).__name__}: {exc}"))
            continue
        skipped.extend((year, rnd, event, drv, note) for drv, note in notes)
        if pace.empty:
            skipped.append((year, rnd, event, None, "fewer than 2 drivers with enough clean laps"))
        else:
            out.append(pace)
    pace = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    skipped = (pd.DataFrame(skipped, columns=["Year", "Round", "Event", "Driver", "Reason"])
               .sort_values(["Year", "Round"]).reset_index(drop=True))
    return pace.sort_values(["Year", "Round", "PacePct"]).reset_index(drop=True), skipped
