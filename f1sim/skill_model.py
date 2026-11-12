"""Hierarchical Bayesian model: driver skill + car pace, from quali and race pace.

For every observation of a driver's pace (in % of a lap; lower = faster):

    pace = segment/race offset + driver skill + car pace + noise

Observations
    Qualifying  each driver's best lap in each segment they took part in (Q1,
                Q2, Q3), as % behind that segment's fastest lap. Each segment of each event
                gets its own offset, so track evolution from Q1 to Q3 isn't
                mistaken for pace.
    Race        each driver's race pace relative to the field, from the
                per-race lap model (race_pace.driver_race_pace). Its standard
                error is added to the noise, so uncertain races count less.
                A scale factor (beta) lets race gaps be compressed or
                stretched relative to one-lap gaps (tyre management, traffic).

Structure
    Driver skill   one value per driver per season; drifts from season to
                   season as a random walk over the driver's active seasons.
                   Centred on each season's field average.
    Car pace       one value per team per race; evolves from race to race
                   within a team-season (upgrades) as a random walk, starting
                   fresh each season. Centred on the field average at each race.
    Noise          Student-t (heavy tails), separate for qualifying and race,
                   so crashes, traffic or ruined laps can't drag estimates.

Teammates share a car at each race, which separates driver skill from car
pace; drivers who change teams link the whole grid together.

Units: % of lap time. Skill -0.3 means 0.3% faster than the season's average
driver in the same car; car +0.5 means 0.5% slower than the race's average car.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

QUALI_SEGMENTS = ["Q1", "Q2", "Q3"]


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
@dataclass
class ModelData:
    quali: pd.DataFrame         # one row per quali lap observation
    race: pd.DataFrame          # one row per driver race-pace observation
    races: pd.DataFrame         # chronological race index
    skill_cells: pd.DataFrame   # (DriverId, Year) cells, grouped by driver
    car_cells: pd.DataFrame     # (TeamId, Year, Round) cells, grouped by team-season
    quali_sessions: pd.DataFrame
    unmatched_race_rows: pd.DataFrame


def _quali_observations(quali: pd.DataFrame) -> pd.DataFrame:
    q = quali.melt(id_vars=["Year", "Round", "Event", "DriverId", "TeamId", "Abbreviation"],
                   value_vars=QUALI_SEGMENTS, var_name="Segment", value_name="Time").dropna(subset=["Time"])
    fastest = q.groupby(["Year", "Round", "Segment"])["Time"].transform("min")
    q["y"] = 100 * (q["Time"] / fastest - 1)
    n = q.groupby(["Year", "Round", "Segment"])["Time"].transform("size")
    return q[n >= 3].reset_index(drop=True)


def _race_observations(race_pace: pd.DataFrame, results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ident = results[["Year", "Round", "Abbreviation", "DriverId", "TeamId"]].drop_duplicates()
    r = race_pace.rename(columns={"Driver": "Abbreviation"}).merge(
        ident, on=["Year", "Round", "Abbreviation"], how="left")
    unmatched = r[r["DriverId"].isna() | r["TeamId"].isna()]
    r = r.dropna(subset=["DriverId", "TeamId"]).reset_index(drop=True)
    r["y"], r["se"] = r["PacePct"], r["PaceSEPct"]
    return r, unmatched


def _cells(df: pd.DataFrame, keys: list[str], group: list[str]) -> pd.DataFrame:
    """Unique cells sorted so each group's cells are consecutive and in time order;
    adds the cell index, the group's first position and an is-first flag."""
    cells = df[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    cells["cell"] = np.arange(len(cells))
    first = cells.groupby(group)["cell"].transform("min")
    cells["first_pos"] = first.to_numpy()
    cells["is_first"] = cells["cell"] == first
    return cells


def prepare_data(quali: pd.DataFrame, race_pace: pd.DataFrame, results: pd.DataFrame,
                 forecast: pd.DataFrame | None = None) -> ModelData:
    """quali: load_quali(); race_pace: driver_race_pace(...)[0]; results: load_results().

    forecast: optional entry lists of future races to predict (DriverId, TeamId,
    Year, Round, Event), with no observations. Their skill and car cells are
    added so the random walks carry drivers and cars forward from the latest
    data: a driver's new season starts from their previous season, a team's
    future races continue from its latest race that season (a season the team
    has no data for yet starts from the prior), and a debutant gets the prior.
    """
    q = _quali_observations(quali)
    r, unmatched = _race_observations(race_pace, results)
    fc = (forecast[["DriverId", "TeamId", "Year", "Round", "Event"]].drop_duplicates()
          if forecast is not None else pd.DataFrame(columns=["DriverId", "TeamId", "Year", "Round", "Event"]))

    races = (pd.concat([q[["Year", "Round", "Event"]], r[["Year", "Round", "Event"]], fc[["Year", "Round", "Event"]]])
             .drop_duplicates(["Year", "Round"]).sort_values(["Year", "Round"]).reset_index(drop=True))
    races["race"] = np.arange(len(races))

    obs = pd.concat([q[["DriverId", "TeamId", "Year", "Round"]], r[["DriverId", "TeamId", "Year", "Round"]],
                     fc[["DriverId", "TeamId", "Year", "Round"]]])
    obs[["Year", "Round"]] = obs[["Year", "Round"]].astype(int)
    skill_cells = _cells(obs, ["DriverId", "Year"], ["DriverId"])
    car_cells = _cells(obs, ["TeamId", "Year", "Round"], ["TeamId", "Year"])
    car_cells = car_cells.merge(races[["Year", "Round", "race"]], on=["Year", "Round"], how="left")
    seasons = sorted(skill_cells["Year"].unique())
    skill_cells["season"] = skill_cells["Year"].map({y: i for i, y in enumerate(seasons)})

    quali_sessions = (q[["Year", "Round", "Segment"]].drop_duplicates()
                      .sort_values(["Year", "Round", "Segment"]).reset_index(drop=True))
    quali_sessions["qsession"] = np.arange(len(quali_sessions))

    def attach(df):
        df = df.merge(skill_cells[["DriverId", "Year", "cell"]].rename(columns={"cell": "skill_idx"}),
                      on=["DriverId", "Year"])
        df = df.merge(car_cells[["TeamId", "Year", "Round", "cell"]].rename(columns={"cell": "car_idx"}),
                      on=["TeamId", "Year", "Round"])
        return df.merge(races[["Year", "Round", "race"]], on=["Year", "Round"])

    q = attach(q).merge(quali_sessions, on=["Year", "Round", "Segment"])
    r = attach(r)
    return ModelData(q, r, races, skill_cells, car_cells, quali_sessions, unmatched)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def _averaging_matrix(group_of_cell: np.ndarray, n_groups: int) -> np.ndarray:
    """Matrix M with (M @ x)[g] = mean of x over the cells in group g."""
    M = np.zeros((n_groups, len(group_of_cell)))
    M[group_of_cell, np.arange(len(group_of_cell))] = 1.0
    return M / M.sum(axis=1, keepdims=True)


def build_model(data: ModelData):
    import pymc as pm
    import pytensor.tensor as pt

    sk, cc = data.skill_cells, data.car_cells
    q, r = data.quali, data.race
    n_sk, n_car = len(sk), len(cc)
    n_seasons = int(sk["season"].max()) + 1
    M_season = _averaging_matrix(sk["season"].to_numpy(), n_seasons)
    M_race = _averaging_matrix(cc["race"].to_numpy(), len(data.races))

    def walk(z, first, start_sd, step_sd, first_pos):
        """Random walk within each group: first cell ~ start_sd, then steps ~ step_sd."""
        steps = pt.where(first, start_sd * z, step_sd * z)
        cs = pt.concatenate([pt.zeros(1), pt.cumsum(steps)])
        return cs[np.arange(len(first_pos)) + 1] - cs[first_pos]

    coords = {"skill_cell": np.arange(n_sk), "car_cell": np.arange(n_car),
              "qsession": np.arange(len(data.quali_sessions)), "race": np.arange(len(data.races))}
    with pm.Model(coords=coords) as model:
        # Driver skill: season-to-season random walk, centred per season
        sigma_skill = pm.HalfNormal("sigma_skill", 0.5)
        tau_skill = pm.HalfNormal("tau_skill", 0.2)
        z_skill = pm.Normal("z_skill", 0, 1, dims="skill_cell")
        skill = walk(z_skill, sk["is_first"].to_numpy(), sigma_skill, tau_skill, sk["first_pos"].to_numpy())
        skill_c = pm.Deterministic(
            "skill", skill - pt.dot(M_season, skill)[sk["season"].to_numpy()], dims="skill_cell")

        # Car pace: race-to-race random walk within each team-season, centred per race
        sigma_car = pm.HalfNormal("sigma_car", 1.0)
        tau_car = pm.HalfNormal("tau_car", 0.1)
        z_car = pm.Normal("z_car", 0, 1, dims="car_cell")
        car = walk(z_car, cc["is_first"].to_numpy(), sigma_car, tau_car, cc["first_pos"].to_numpy())
        car_c = pm.Deterministic(
            "car", car - pt.dot(M_race, car)[cc["race"].to_numpy()], dims="car_cell")

        # Qualifying: best lap per driver per segment
        alpha_q = pm.Normal("alpha_q", 1.0, 2.0, dims="qsession")
        sigma_q = pm.HalfNormal("sigma_q", 0.5)
        nu_q = pm.Gamma("nu_q", 2.0, 0.1)
        mu_q = (alpha_q[q["qsession"].to_numpy()] + skill_c[q["skill_idx"].to_numpy()]
                + car_c[q["car_idx"].to_numpy()])
        pm.StudentT("y_quali", nu=nu_q, mu=mu_q, sigma=sigma_q, observed=q["y"].to_numpy())

        # Race pace (already relative to the field, with known standard errors)
        if len(r):
            alpha_r = pm.Normal("alpha_r", 0.0, 1.0, dims="race")
            beta = pm.Normal("beta_race", 1.0, 0.3)
            sigma_r = pm.HalfNormal("sigma_race", 0.5)
            nu_r = pm.Gamma("nu_race", 2.0, 0.1)
            mu_r = (alpha_r[r["race"].to_numpy()]
                    + beta * (skill_c[r["skill_idx"].to_numpy()] + car_c[r["car_idx"].to_numpy()]))
            sd_r = pt.sqrt(r["se"].to_numpy() ** 2 + sigma_r ** 2)
            pm.StudentT("y_race", nu=nu_r, mu=mu_r, sigma=sd_r, observed=r["y"].to_numpy())
    return model


def fit(model, draws: int = 1000, tune: int = 1000, chains: int = 2,
        target_accept: float = 0.9, seed: int = 0, nuts_sampler: str = "pymc"):
    import pymc as pm
    with model:
        return pm.sample(draws=draws, tune=tune, chains=chains, cores=chains,
                         target_accept=target_accept, random_seed=seed,
                         nuts_sampler=nuts_sampler)


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
HYPER = ["sigma_skill", "tau_skill", "sigma_car", "tau_car", "sigma_q", "nu_q",
         "beta_race", "sigma_race", "nu_race"]


def _draws(idata, name: str) -> np.ndarray:
    """Posterior draws as (n_draws, ...) for InferenceData or DataTree."""
    v = np.asarray(idata.posterior[name].values)
    return v.reshape(-1, *v.shape[2:])


def diagnostics(idata) -> dict:
    import arviz as az
    present = [h for h in HYPER if h in idata.posterior]
    s = az.summary(idata, var_names=present + ["skill", "car"])
    return {
        "divergences": int(np.asarray(idata.sample_stats["diverging"].values).sum()),
        "max_r_hat": float(pd.to_numeric(s["r_hat"], errors="coerce").max()),
        "min_ess_bulk": float(pd.to_numeric(s["ess_bulk"], errors="coerce").min()),
        "hyperparameters": az.summary(idata, var_names=present),
    }


def _interval(draws: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"Mean": draws.mean(axis=0),
                         "Low90": np.percentile(draws, 5, axis=0),
                         "High90": np.percentile(draws, 95, axis=0)})


def skill_table(idata, data: ModelData, names: pd.DataFrame | None = None) -> pd.DataFrame:
    """Driver skill per season (% of lap vs the season's average driver; negative = faster)."""
    out = pd.concat([data.skill_cells[["DriverId", "Year"]], _interval(_draws(idata, "skill"))], axis=1)
    n_q = data.quali.groupby(["DriverId", "Year"])["Round"].nunique().rename("QualiEvents")
    n_r = data.race.groupby(["DriverId", "Year"]).size().rename("RaceObs")
    out = out.join(n_q, on=["DriverId", "Year"]).join(n_r, on=["DriverId", "Year"])
    if names is not None:
        out = out.merge(names, on="DriverId", how="left")
    return out.sort_values(["Year", "Mean"]).reset_index(drop=True)


def car_table(idata, data: ModelData, names: pd.DataFrame | None = None) -> pd.DataFrame:
    """Car pace per team per race (% vs the race's average car; negative = faster)."""
    out = pd.concat([data.car_cells[["TeamId", "Year", "Round"]], _interval(_draws(idata, "car"))], axis=1)
    out = out.merge(data.races[["Year", "Round", "Event"]], on=["Year", "Round"], how="left")
    if names is not None:
        out = out.merge(names, on=["TeamId", "Year"], how="left")
    return out.sort_values(["Year", "Round", "Mean"]).reset_index(drop=True)


def car_season_table(idata, data: ModelData, names: pd.DataFrame | None = None) -> pd.DataFrame:
    """Season-average car pace per team, with uncertainty over the whole season."""
    draws = _draws(idata, "car")
    cc = data.car_cells
    rows = []
    for (team, year), idx in cc.groupby(["TeamId", "Year"]).groups.items():
        season = draws[:, np.asarray(idx)].mean(axis=1)
        rows.append({"TeamId": team, "Year": year, "Mean": season.mean(),
                     "Low90": np.percentile(season, 5), "High90": np.percentile(season, 95),
                     "Races": len(idx)})
    out = pd.DataFrame(rows)
    if names is not None:
        out = out.merge(names, on=["TeamId", "Year"], how="left")
    return out.sort_values(["Year", "Mean"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Saving and loading a fit (to Google Drive, for the simulator later)
# ----------------------------------------------------------------------------
def save_fit(idata, data: ModelData, paths, name: str = "driver_car_model") -> dict:
    """Save posterior draws (.nc) and the data/index tables (.pkl) under paths.models_dir."""
    import pickle
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    nc, pkl = paths.models_dir / f"{name}.nc", paths.models_dir / f"{name}_data.pkl"
    errors = []
    for engine in ("h5netcdf", "netcdf4"):
        try:
            idata.to_netcdf(str(nc), engine=engine)
            break
        except (ValueError, ImportError, TypeError) as exc:
            errors.append(f"{engine}: {exc}")
    else:
        raise RuntimeError("Couldn't save the fit. Install a NetCDF backend: "
                           "`pip install h5netcdf h5py`.\n" + "\n".join(errors))
    with open(pkl, "wb") as f:
        pickle.dump(data, f)
    return {"posterior": nc, "data": pkl}


def load_fit(paths, name: str = "driver_car_model"):
    """Load a fit saved by save_fit. Returns (idata, data)."""
    import pickle
    import arviz as az
    nc, pkl = paths.models_dir / f"{name}.nc", paths.models_dir / f"{name}_data.pkl"
    try:
        idata = az.from_netcdf(str(nc), engine="h5netcdf")
    except TypeError:  # older/newer ArviZ without the engine argument
        idata = az.from_netcdf(str(nc))
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return idata, data
