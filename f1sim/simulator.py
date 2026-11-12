"""Monte Carlo race simulator (dry races).

Each simulated race:
1. Draw one sample from the driver/car model's posterior (skill, car pace,
   noise levels), so uncertainty about skill and cars flows into the results.
2. Qualifying: one-lap pace = driver skill + car pace at this circuit, plus
   qualifying noise (heavy-tailed, as fitted). The order sets the grid.
3. Retirements: each car retires with its team-season's DNF rate, adjusted for
   the circuit's DNF rate (both from dry races, shrunk toward the average when
   there's little data). Retired cars drop to the back.
4. Race: race pace = beta x (skill + car) plus race-day noise. Starting from
   the grid, a car passes the car ahead only when it's faster by more than the
   circuit's overtaking threshold, so a fast car can stay stuck at Monaco. The
   threshold for each circuit is calibrated so simulated position changes
   match that circuit's real dry races (calibrate_overtaking).
5. Points: 25-18-15-12-10-8-6-4-2-1 for the top 10 finishers (no fastest-lap
   point, as in 2025).

Choices you control: the field (any driver with a season they raced in
2018-2025, in any team-season's car), the circuit, and the number of
simulations. Car pace at a circuit uses that team's value at that circuit in
that season; if the team didn't race there that season, its season average.

Caveat: for a real driver-car pairing, skill + car is well measured. For a
swap (a driver in a car they never drove), the result depends on how the model
split skill from car, which teammate comparisons pin down only partly; the
intervals don't capture all of that. Swapped entries are flagged.
"""
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1sim.baseline import is_finished
from f1sim.skill_model import ModelData, _draws

POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)
DNF_PACE = 1e6  # retired cars: "infinitely slow", so they sink to the back


# ----------------------------------------------------------------------------
# Posterior and lookups
# ----------------------------------------------------------------------------
@dataclass
class Posterior:
    skill: np.ndarray       # (draws, skill cells)
    car: np.ndarray         # (draws, car cells)
    beta: np.ndarray        # (draws,)
    sigma_race: np.ndarray
    nu_race: np.ndarray
    sigma_q: np.ndarray
    nu_q: np.ndarray
    data: ModelData

    @property
    def n_draws(self) -> int:
        return len(self.beta)


def load_posterior(idata, data: ModelData) -> Posterior:
    return Posterior(
        skill=_draws(idata, "skill"), car=_draws(idata, "car"),
        beta=_draws(idata, "beta_race"), sigma_race=_draws(idata, "sigma_race"),
        nu_race=_draws(idata, "nu_race"), sigma_q=_draws(idata, "sigma_q"),
        nu_q=_draws(idata, "nu_q"), data=data,
    )


def calendar(results: pd.DataFrame) -> pd.DataFrame:
    """One row per race with its circuit."""
    if "CircuitId" not in results.columns or results["CircuitId"].isna().any():
        raise ValueError("Race results have no CircuitId. Rerun notebook 05 with FORCE ticked "
                         "to re-download results with circuit information.")
    return (results[["Year", "Round", "Event", "CircuitId", "CircuitName"]]
            .drop_duplicates(["Year", "Round"]).sort_values(["Year", "Round"]).reset_index(drop=True))


def _skill_index(post: Posterior) -> dict:
    sk = post.data.skill_cells
    return {(d, int(y)): int(c) for d, y, c in zip(sk["DriverId"], sk["Year"], sk["cell"])}


def _car_cells_for(post: Posterior, cal: pd.DataFrame, team: str, year: int,
                   circuit: str) -> tuple[list[int], str]:
    """Car cells to average for a team-season at a circuit, and a description."""
    cc = post.data.car_cells
    season = cc[(cc["TeamId"] == team) & (cc["Year"] == year)]
    if season.empty:
        raise KeyError(f"No car for team '{team}' in {year}.")
    rounds = set(cal.loc[(cal["Year"] == year) & (cal["CircuitId"] == circuit), "Round"])
    here = season[season["Round"].isin(rounds)]
    if len(here):
        return here["cell"].astype(int).tolist(), "at this circuit"
    return season["cell"].astype(int).tolist(), "season average (didn't race here that year)"


# ----------------------------------------------------------------------------
# DNF rates
# ----------------------------------------------------------------------------
@dataclass
class DNFRates:
    overall: float
    team: pd.DataFrame      # TeamId, Year, Starts, DNFs, Rate (shrunk)
    circuit: pd.DataFrame   # CircuitId, Entries, DNFs, Rate (shrunk)

    def probability(self, team: str, year: int, circuit: str) -> float:
        t = self.team.loc[(self.team["TeamId"] == team) & (self.team["Year"] == year), "Rate"]
        c = self.circuit.loc[self.circuit["CircuitId"] == circuit, "Rate"]
        t = float(t.iloc[0]) if len(t) else self.overall
        c = float(c.iloc[0]) if len(c) else self.overall
        return float(np.clip(t * c / self.overall, 0.005, 0.5))


def dnf_rates(results: pd.DataFrame, dry_keys: set | None = None,
              k_team: float = 20, k_circuit: float = 40) -> DNFRates:
    """DNF rates from dry races. k_* = how many average entries the shrinkage
    toward the overall rate is worth (more data -> closer to the raw rate)."""
    r = results.copy()
    if dry_keys is not None:
        r = r[[(y, n) in dry_keys for y, n in zip(r["Year"], r["Round"])]]
    r["DNF"] = (~r["Status"].map(is_finished)).astype(int)
    overall = float(r["DNF"].mean())

    team = r.groupby(["TeamId", "Year"]).agg(Starts=("DNF", "size"), DNFs=("DNF", "sum")).reset_index()
    team["Rate"] = (team["DNFs"] + k_team * overall) / (team["Starts"] + k_team)
    circ = r.groupby("CircuitId").agg(Entries=("DNF", "size"), DNFs=("DNF", "sum")).reset_index()
    circ["Rate"] = (circ["DNFs"] + k_circuit * overall) / (circ["Entries"] + k_circuit)
    return DNFRates(overall, team, circ)


# ----------------------------------------------------------------------------
# Race mechanics
# ----------------------------------------------------------------------------
def overtake(pace_in_grid_order: np.ndarray, threshold) -> np.ndarray:
    """Finishing order from grid order: a car passes the car directly ahead
    when it's faster by more than `threshold` (same units as pace; lower =
    faster). Repeats until nobody can pass.

    pace_in_grid_order: (sims, cars), column 0 = pole. threshold: scalar or
    (sims, 1). Returns (sims, cars): for each finishing position, the grid
    slot of the car in it. Always terminates: each pass removes one
    out-of-pace-order pair and no pass can be undone.
    """
    p = np.array(pace_in_grid_order, dtype=float, copy=True)
    S, N = p.shape
    order = np.tile(np.arange(N), (S, 1))
    for _ in range(N * N):
        moved = False
        for start in (0, 1):
            a = np.arange(start, N - 1, 2)
            if not len(a):
                continue
            ahead, behind = p[:, a], p[:, a + 1]
            swap = (ahead - behind) > threshold
            if swap.any():
                moved = True
                p[:, a], p[:, a + 1] = np.where(swap, behind, ahead), np.where(swap, ahead, behind)
                oa, ob = order[:, a], order[:, a + 1]
                order[:, a], order[:, a + 1] = np.where(swap, ob, oa), np.where(swap, oa, ob)
        if not moved:
            break
    return order


def _student_t(rng, nu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
    """Student-t noise, one (nu, sigma) per simulation row."""
    return sigma[:, None] * rng.standard_t(nu[:, None], size=(len(nu), n))


# ----------------------------------------------------------------------------
# Calibrating overtaking difficulty per circuit
# ----------------------------------------------------------------------------
def _race_field_pace(post: Posterior, race: pd.DataFrame, skill_idx: dict,
                     car_idx: dict) -> pd.DataFrame:
    """Finishers of a real race with grid rank, finish rank and model cells."""
    fin = race[race["Status"].map(is_finished)].copy()
    fin["skill_cell"] = [skill_idx.get((d, int(y))) for d, y in zip(fin["DriverId"], fin["Year"])]
    fin["car_cell"] = [car_idx.get((t, int(y), int(n)))
                       for t, y, n in zip(fin["TeamId"], fin["Year"], fin["Round"])]
    fin = fin.dropna(subset=["skill_cell", "car_cell"])
    grid = pd.to_numeric(fin["GridPosition"], errors="coerce").astype(float)
    fin["GridSort"] = grid.where(grid > 0, 99)  # pit-lane start = last
    fin = fin.sort_values(["GridSort", "Position"]).reset_index(drop=True)
    fin["GridRank"] = np.arange(1, len(fin) + 1)
    fin["FinishRank"] = fin["Position"].rank(method="first").astype(int)
    return fin


def calibrate_overtaking(results: pd.DataFrame, post: Posterior, dry_keys: set,
                         thresholds=np.round(np.arange(0.0, 2.01, 0.05), 2),
                         n_sims: int = 200, k_shrink: float = 2.0, seed: int = 0) -> pd.DataFrame:
    """Per circuit: the overtaking threshold (% of lap) whose simulated mean
    position change among finishers matches that circuit's real dry races.

    Each circuit's threshold is shrunk toward the all-circuit threshold by
    k_shrink races' worth, so a circuit with one or two races isn't fitted to
    noise. RawAtLimit = even the extreme threshold couldn't match the real
    races (e.g. more shuffling than race pace alone explains: strategy, safety
    cars). Uses the real grid and finishers of each race, and model race pace.
    """
    rng = np.random.default_rng(seed)
    cal = calendar(results)
    skill_idx = _skill_index(post)
    cc = post.data.car_cells
    car_idx = {(t, int(y), int(n)): int(c) for t, y, n, c in zip(cc["TeamId"], cc["Year"], cc["Round"], cc["cell"])}
    T = np.asarray(thresholds, dtype=float)

    rows = []
    for (year, rnd), race in results.groupby(["Year", "Round"]):
        if (year, rnd) not in dry_keys:
            continue
        fin = _race_field_pace(post, race, skill_idx, car_idx)
        if len(fin) < 6:
            continue
        observed = float((fin["FinishRank"] - fin["GridRank"]).abs().mean())
        d = rng.integers(post.n_draws, size=n_sims)
        base = post.skill[d][:, fin["skill_cell"].astype(int)] + post.car[d][:, fin["car_cell"].astype(int)]
        pace = post.beta[d][:, None] * base + _student_t(rng, post.nu_race[d], post.sigma_race[d], len(fin))
        # every threshold on the same simulated paces (common random numbers)
        pace_all = np.tile(pace, (len(T), 1))
        thr = np.repeat(T, n_sims)[:, None]
        finish_slots = overtake(pace_all, thr)               # grid slot per finishing position
        finish_rank = np.argsort(finish_slots, axis=1) + 1   # finishing position per grid slot
        change = np.abs(finish_rank - np.arange(1, len(fin) + 1)).mean(axis=1)
        curve = change.reshape(len(T), n_sims).mean(axis=1)
        circuit = cal.loc[(cal["Year"] == year) & (cal["Round"] == rnd), "CircuitId"].iloc[0]
        rows.append({"CircuitId": circuit, "Year": year, "Round": rnd,
                     "Observed": observed, **{f"t{t:.2f}": v for t, v in zip(T, curve)}})

    per_race = pd.DataFrame(rows)
    tcols = [f"t{t:.2f}" for t in T]

    def crossing(curve, target):
        """Threshold where the (decreasing) simulated curve meets the target."""
        return float(np.interp(target, curve[::-1], T[::-1]))  # np.interp needs increasing x

    pooled = crossing(per_race[tcols].mean().to_numpy(), per_race["Observed"].mean())
    out = []
    for circuit, g in per_race.groupby("CircuitId"):
        n, obs = len(g), g["Observed"].mean()
        curve = g[tcols].mean().to_numpy()
        raw = crossing(curve, obs)
        # Shrink the threshold itself toward the all-circuit value, so circuits
        # with one or two races (or unusual field sizes) aren't fitted to noise
        thr = (n * raw + k_shrink * pooled) / (n + k_shrink)
        out.append({"CircuitId": circuit, "DryRaces": n, "ObservedPosChange": round(obs, 2),
                    "RawThreshold": round(raw, 3), "Threshold": round(thr, 3),
                    "SimPosChange": round(float(np.interp(thr, T, curve)), 2),
                    "RawAtLimit": raw <= T[0] or raw >= T[-1]})
    names = cal.drop_duplicates("CircuitId").set_index("CircuitId")["CircuitName"]
    out = pd.DataFrame(out)
    out.insert(1, "CircuitName", out["CircuitId"].map(names))
    return out.sort_values("Threshold", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Building a field
# ----------------------------------------------------------------------------
def driver_lookup(results: pd.DataFrame) -> pd.DataFrame:
    """DriverId, Abbreviation, FullName (latest spelling)."""
    return (results.sort_values(["Year", "Round"])
            .groupby("DriverId")[["Abbreviation", "FullName"]].last().reset_index())


def team_lookup(results: pd.DataFrame) -> pd.DataFrame:
    """TeamId, Year, TeamName (most common name that season)."""
    return (results.groupby(["TeamId", "Year"])["TeamName"]
            .agg(lambda s: s.value_counts().index[0]).reset_index())


def resolve_driver(results: pd.DataFrame, key: str) -> str:
    """3-letter code (e.g. 'HAM') or Jolpica id (e.g. 'hamilton') -> DriverId."""
    drivers = driver_lookup(results)
    if key in set(drivers["DriverId"]):
        return key
    hits = drivers[drivers["Abbreviation"].str.upper() == key.strip().upper()]
    if len(hits) == 1:
        return hits["DriverId"].iloc[0]
    if len(hits) > 1:
        raise KeyError(f"'{key}' matches several drivers: {hits['DriverId'].tolist()}. Use the id.")
    raise KeyError(f"No driver '{key}'. Use a 3-letter code such as HAM or VER.")


def resolve_team(results: pd.DataFrame, key: str, year: int) -> str:
    """TeamId (e.g. 'red_bull') or team name (e.g. 'Red Bull Racing') -> TeamId."""
    teams = team_lookup(results)
    season = teams[teams["Year"] == year]
    if season.empty:
        raise KeyError(f"No data for {year}. Seasons available: {sorted(teams['Year'].unique())}")
    k = key.strip().lower()
    if k in set(season["TeamId"]):
        return k
    hits = season[season["TeamName"].str.lower() == k]
    if len(hits) == 1:
        return hits["TeamId"].iloc[0]
    raise KeyError(f"No team '{key}' in {year}. Options: "
                   + ", ".join(f"{i} ({n})" for i, n in zip(season["TeamId"], season["TeamName"])))


def latest_skill_year(post: Posterior, driver_id: str) -> int:
    years = post.data.skill_cells.loc[post.data.skill_cells["DriverId"] == driver_id, "Year"]
    if years.empty:
        raise KeyError(f"No skill estimate for '{driver_id}'.")
    return int(years.max())


def field_from_race(results: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame:
    """The real field of a race: each driver in their own car that season."""
    race = results[(results["Year"] == year) & (results["Round"] == rnd)]
    if race.empty:
        raise KeyError(f"No race {year} round {rnd}.")
    return pd.DataFrame({
        "DriverId": race["DriverId"].to_numpy(), "SkillYear": year,
        "TeamId": race["TeamId"].to_numpy(), "CarYear": year,
    }).reset_index(drop=True)


_SWAP = re.compile(r"^\s*([^=@]+?)\s*=\s*([^=@]+?)\s*(?:@\s*(\d{4}))?\s*$")


def apply_driver_swaps(field: pd.DataFrame, results: pd.DataFrame, post: Posterior, spec: str) -> pd.DataFrame:
    """'TSU=HAM@2019; LAW=VET' -> replace TSU with Hamilton (2019 skill) and
    LAW with Vettel (latest season). The new driver takes the old one's car."""
    field = field.copy()
    for part in filter(str.strip, spec.split(";")):
        m = _SWAP.match(part)
        if not m:
            raise ValueError(f"Can't read '{part}'. Use OLD=NEW or OLD=NEW@YEAR, e.g. TSU=HAM@2019.")
        old, new, year = resolve_driver(results, m[1]), resolve_driver(results, m[2]), m[3]
        rows = field.index[field["DriverId"] == old]
        if not len(rows):
            raise KeyError(f"'{m[1]}' isn't in this field.")
        field.loc[rows, "DriverId"] = new
        field.loc[rows, "SkillYear"] = int(year) if year else latest_skill_year(post, new)
    return field


def apply_car_changes(field: pd.DataFrame, results: pd.DataFrame, spec: str) -> pd.DataFrame:
    """'HAM=ferrari@2024; VER=mclaren' -> put HAM in the 2024 Ferrari and VER in
    the McLaren of his current car season."""
    field = field.copy()
    for part in filter(str.strip, spec.split(";")):
        m = _SWAP.match(part)
        if not m:
            raise ValueError(f"Can't read '{part}'. Use DRIVER=TEAM or DRIVER=TEAM@YEAR, e.g. HAM=ferrari@2024.")
        driver = resolve_driver(results, m[1])
        rows = field.index[field["DriverId"] == driver]
        if not len(rows):
            raise KeyError(f"'{m[1]}' isn't in this field.")
        year = int(m[3]) if m[3] else int(field.loc[rows[0], "CarYear"])
        field.loc[rows, "TeamId"] = resolve_team(results, m[2], year)
        field.loc[rows, "CarYear"] = year
    return field


def describe_field(field: pd.DataFrame, results: pd.DataFrame, post: Posterior,
                   circuit: str) -> pd.DataFrame:
    """The field with names, where each car value comes from and swap flags.
    Raises if a driver has no skill estimate for the chosen season."""
    cal = calendar(results)
    skill_idx = _skill_index(post)
    real = set(zip(results["DriverId"], results["TeamId"], results["Year"]))
    names, teams = driver_lookup(results).set_index("DriverId"), team_lookup(results)
    rows = []
    for e in field.itertuples(index=False):
        if (e.DriverId, int(e.SkillYear)) not in skill_idx:
            seasons = sorted(post.data.skill_cells.loc[post.data.skill_cells["DriverId"] == e.DriverId, "Year"])
            raise KeyError(f"No skill estimate for {e.DriverId} in {e.SkillYear}. Seasons available: {seasons}")
        cells, source = _car_cells_for(post, cal, e.TeamId, int(e.CarYear), circuit)
        tn = teams.loc[(teams["TeamId"] == e.TeamId) & (teams["Year"] == e.CarYear), "TeamName"]
        rows.append({
            "Driver": names.loc[e.DriverId, "Abbreviation"], "FullName": names.loc[e.DriverId, "FullName"],
            "DriverId": e.DriverId, "SkillYear": int(e.SkillYear),
            "Team": tn.iloc[0] if len(tn) else e.TeamId, "TeamId": e.TeamId, "CarYear": int(e.CarYear),
            "CarValue": source,
            "Swapped": (e.DriverId, e.TeamId, int(e.CarYear)) not in real or int(e.SkillYear) != int(e.CarYear),
            "_skill_cell": skill_idx[(e.DriverId, int(e.SkillYear))], "_car_cells": cells,
        })
    out = pd.DataFrame(rows)
    if out["DriverId"].duplicated().any():
        dup = out.loc[out["DriverId"].duplicated(), "Driver"].tolist()
        raise ValueError(f"{dup} would appear twice. They're already in this field: swap them "
                         "out too, or pick a driver who isn't on this grid.")
    return out


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
@dataclass
class SimResult:
    entries: pd.DataFrame       # describe_field output
    grid: np.ndarray            # (sims, cars) grid position per car
    finish: np.ndarray          # (sims, cars) classified position, 0 = DNF
    points: np.ndarray          # (sims, cars)
    circuit: str
    threshold: float
    dnf_prob: np.ndarray        # (cars,)

    def summary(self) -> pd.DataFrame:
        f = self.finish
        s = self.entries[["Driver", "FullName", "SkillYear", "Team", "CarYear", "Swapped"]].copy()
        s["Win%"] = 100 * (f == 1).mean(axis=0)
        s["Podium%"] = 100 * ((f >= 1) & (f <= 3)).mean(axis=0)
        s["Points%"] = 100 * ((f >= 1) & (f <= 10)).mean(axis=0)
        s["DNF%"] = 100 * (f == 0).mean(axis=0)
        s["Pole%"] = 100 * (self.grid == 1).mean(axis=0)
        s["AvgGrid"] = self.grid.mean(axis=0)
        with np.errstate(invalid="ignore"):
            s["AvgFinish"] = np.nanmean(np.where(f > 0, f, np.nan), axis=0)  # when finishing
        s["ExpPoints"] = self.points.mean(axis=0)
        s["ExpPointsLow90"] = np.percentile(self.points, 5, axis=0)
        s["ExpPointsHigh90"] = np.percentile(self.points, 95, axis=0)
        return s.sort_values(["ExpPoints", "Win%"], ascending=False).round(2).reset_index(drop=True)

    def position_probabilities(self) -> pd.DataFrame:
        """Rows = drivers, columns = P1..Pn and DNF, values = % of simulations."""
        n = self.finish.shape[1]
        cols = {f"P{k}": 100 * (self.finish == k).mean(axis=0) for k in range(1, n + 1)}
        cols["DNF"] = 100 * (self.finish == 0).mean(axis=0)
        return pd.DataFrame(cols, index=self.entries["Driver"]).round(2)

    def team_points(self) -> pd.DataFrame:
        """Expected points per team (sum of its cars), with a 90% interval."""
        teams = self.entries["Team"] + " " + self.entries["CarYear"].astype(str)
        rows = []
        for t in teams.unique():
            pts = self.points[:, (teams == t).to_numpy()].sum(axis=1)
            rows.append({"Team": t, "Cars": int((teams == t).sum()),
                         "Drivers": ", ".join(self.entries.loc[teams == t, "Driver"]),
                         "ExpPoints": pts.mean(), "Low90": np.percentile(pts, 5),
                         "High90": np.percentile(pts, 95)})
        return pd.DataFrame(rows).sort_values("ExpPoints", ascending=False).round(2).reset_index(drop=True)


def simulate(entries: pd.DataFrame, post: Posterior, circuit: str, threshold: float,
             dnf: DNFRates | None, n_sims: int = 5000, seed: int = 0) -> SimResult:
    """Simulate a race for a described field (describe_field) at a circuit.
    dnf=None: nobody retires."""
    rng = np.random.default_rng(seed)
    n = len(entries)
    d = rng.integers(post.n_draws, size=n_sims)

    skill = post.skill[d][:, entries["_skill_cell"].to_numpy()]
    car = np.column_stack([post.car[d][:, cells].mean(axis=1) for cells in entries["_car_cells"]])
    base = skill + car                                                   # one-lap pace, % of lap

    quali = base + _student_t(rng, post.nu_q[d], post.sigma_q[d], n)
    grid_slots = np.argsort(quali, axis=1)                               # car in each grid slot
    grid_pos = np.argsort(grid_slots, axis=1) + 1                        # grid position per car

    race = post.beta[d][:, None] * base + _student_t(rng, post.nu_race[d], post.sigma_race[d], n)
    if dnf is not None:
        p = np.array([dnf.probability(t, y, circuit) for t, y in zip(entries["TeamId"], entries["CarYear"])])
        retired = rng.random((n_sims, n)) < p
    else:
        p, retired = np.zeros(n), np.zeros((n_sims, n), bool)
    race = np.where(retired, DNF_PACE, race)

    pace_grid_order = np.take_along_axis(race, grid_slots, axis=1)
    finish_slots = overtake(pace_grid_order, threshold)                  # grid slot per finishing position
    cars_in_order = np.take_along_axis(grid_slots, finish_slots, axis=1)  # car per finishing position
    finish = np.empty_like(cars_in_order)
    np.put_along_axis(finish, cars_in_order, np.arange(1, n + 1)[None, :].repeat(n_sims, 0), axis=1)
    finish = np.where(retired, 0, finish)                                 # retired cars are all at the back

    points = np.zeros((n_sims, n))
    for k, pts in enumerate(POINTS, start=1):
        points[finish == k] = pts
    return SimResult(entries, grid_pos, finish, points, circuit, threshold, p)
