"""Out-of-sample test of the race simulator, with checkpoint refits.

For each checkpoint round k of the test season (default 2025: rounds 7, 13, 19):
1. Fit the driver/car model on everything before round k (all earlier seasons
   plus the test season's rounds 1..k-1), with the entry lists of the next
   block of races added as forecast cells: drivers and cars are carried
   forward from their latest data by the model's random walks.
2. Calibrate overtaking thresholds on dry races before round k only.
3. For each dry race in the block, simulate the race from its real grid, using
   the actual finishers (no DNFs), and rank drivers by mean simulated finish.
   Twice: with the model's forecast pace as is, and with that pace updated by
   the weekend's qualifying times (simulator.weekend_update), which are
   known before the race.

This matches how the baseline model (notebook 06) was scored: the grid is
known and the finishing order among finishers is predicted, so both use the
same metrics on the same drivers. Note: which races count as "dry" uses the
tyres actually raced; it selects which races to test, it isn't a model input.

Fits are saved to Drive per checkpoint and reused on reruns, so a Colab
disconnect doesn't lose finished checkpoints.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from f1sim import simulator as sim
from f1sim import skill_model as sm
from f1sim.baseline import is_finished

DEFAULT_CHECKPOINTS = (7, 13, 19)


def _before(df: pd.DataFrame, year: int, rnd: int) -> pd.DataFrame:
    return df[(df["Year"] < year) | ((df["Year"] == year) & (df["Round"] < rnd))]


def blocks(results: pd.DataFrame, year: int, checkpoints=DEFAULT_CHECKPOINTS) -> list[tuple[int, int]]:
    """[(start, end_exclusive)] rounds covered by each checkpoint."""
    last = int(results.loc[results["Year"] == year, "Round"].max())
    cps = sorted(checkpoints)
    return [(s, (cps[i + 1] if i + 1 < len(cps) else last + 1)) for i, s in enumerate(cps)]


def fit_checkpoint(quali, race_pace, results, year: int, start: int, end: int, paths,
                   draws=1000, tune=1000, chains=2, refit=False, seed=0):
    """Fit (or load) the model on data before `start`, forecasting rounds start..end-1."""
    name = f"backtest_{year}_r{start:02d}"
    saved = paths.models_dir / f"{name}.nc"
    if saved.exists() and not refit:
        idata, data = sm.load_fit(paths, name)
        return idata, data, "loaded"
    block = results[(results["Year"] == year) & (results["Round"] >= start) & (results["Round"] < end)]
    forecast = block[["DriverId", "TeamId", "Year", "Round", "Event"]]
    data = sm.prepare_data(_before(quali, year, start), _before(race_pace, year, start),
                           _before(results, year, start), forecast=forecast)
    idata = sm.fit(sm.build_model(data), draws=draws, tune=tune, chains=chains, seed=seed)
    sm.save_fit(idata, data, paths, name)
    return idata, data, "fitted"


def _finishers_in_grid_order(race: pd.DataFrame) -> pd.DataFrame:
    fin = race[race["Status"].map(is_finished)].copy()
    grid = pd.to_numeric(fin["GridPosition"], errors="coerce").astype(float)
    fin["GridSort"] = grid.where(grid > 0, 99)  # pit-lane start = last
    fin = fin.sort_values(["GridSort", "Position"]).reset_index(drop=True)
    fin["GridRank"] = np.arange(1, len(fin) + 1)
    fin["FinishRank"] = fin["Position"].rank(method="first").astype(int)
    return fin


def score_block(post, results, dry_keys: set, year: int, start: int, end: int,
                n_sims: int = 3000, seed: int = 0,
                quali: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate each dry race in the block from its real grid.

    quali: load_quali(); if given, also simulate with pace updated by each
    race's qualifying times. Returns (per-driver predictions, thresholds used)."""
    dry_before = {k for k in dry_keys if k[0] < year or (k[0] == year and k[1] < start)}
    thresholds = sim.calibrate_overtaking(_before(results, year, start), post, dry_before)
    thr = thresholds.set_index("CircuitId")["Threshold"]
    cal = sim.calendar(results)

    rows = []
    for rnd in range(start, end):
        if (year, rnd) not in dry_keys:
            continue
        race = results[(results["Year"] == year) & (results["Round"] == rnd)]
        if race.empty:
            continue
        circuit = cal.loc[(cal["Year"] == year) & (cal["Round"] == rnd), "CircuitId"].iloc[0]
        fin = _finishers_in_grid_order(race)
        field = pd.DataFrame({"DriverId": fin["DriverId"], "SkillYear": year,
                              "TeamId": fin["TeamId"], "CarYear": year})
        entries = sim.describe_field(field, results, post, circuit)
        t = float(thr.get(circuit, thr.median()))
        out = fin[["Year", "Round", "Event", "Abbreviation", "GridRank", "FinishRank"]].copy()
        out["CircuitId"], out["Threshold"] = circuit, t
        variants = {"Simulator": None}
        if quali is not None:
            q_race = quali[(quali["Year"] == year) & (quali["Round"] == rnd)]
            pace, info = sim.weekend_update(entries, post, q_race)
            variants["Simulator + weekend quali"] = pace
            out["QualiTimesUsed"], out["QualiDownWeighted"] = info["QualiTimes"], info["DownWeighted"]
        for label, pace in variants.items():
            res = sim.simulate(entries, post, circuit, t, dnf=None, n_sims=n_sims,
                               seed=seed + rnd, fixed_grid=True, pace=pace)
            avg = res.finish.mean(axis=0)
            out[f"Pred ({label})"] = pd.Series(avg).rank(method="first").astype(int).to_numpy()
            out[f"WinProb ({label})"] = (res.finish == 1).mean(axis=0)
        rows.append(out)
    return pd.concat(rows, ignore_index=True), thresholds


def position_scores(df: pd.DataFrame, pred_col: str) -> dict:
    """Mean per-race Spearman, mean absolute position error, % exactly right."""
    per_race = [spearmanr(g["FinishRank"], g[pred_col]).statistic for _, g in df.groupby(["Year", "Round"])]
    err = (df[pred_col] - df["FinishRank"]).abs()
    return {"MeanSpearman": float(np.nanmean(per_race)), "MAE_positions": float(err.mean()),
            "ExactPct": float(100 * (err == 0).mean()), "Races": df.groupby(["Year", "Round"]).ngroups,
            "DriversScored": int(len(df))}


def compare(sim_preds: pd.DataFrame, baseline_preds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score the simulator, grid order and (optionally) the baseline models on
    exactly the same drivers. baseline_preds: fit_evaluate(...)['position_test_pred']."""
    df = sim_preds.rename(columns={"GridRank": "Pred (Grid order)"})
    cols = ["Pred (Grid order)"] + [c for c in sim_preds.columns if c.startswith("Pred (Simulator")]
    if baseline_preds is not None:
        b = baseline_preds.drop(columns=["Pred (Grid order)", "FinishRank", "Event", "TeamName", "Grid"],
                                errors="ignore")
        bcols = [c for c in b.columns if c.startswith("Pred (")]
        df = df.merge(b, on=["Year", "Round", "Abbreviation"], how="inner")
        df = df.dropna(subset=bcols)
        # re-rank baseline predictions among the drivers scored here
        for c in bcols:
            df[c] = df.groupby(["Year", "Round"])[c].rank(method="first").astype(int)
        cols += bcols
    table = pd.DataFrame({c.removeprefix("Pred (").removesuffix(")"): position_scores(df, c) for c in cols}).T
    return table.round(3)


def winner_scores(sim_preds: pd.DataFrame) -> pd.DataFrame:
    """How well each simulator variant's win probabilities picked the winners."""
    rows = {}
    w = sim_preds[sim_preds["FinishRank"] == 1]
    for col in [c for c in sim_preds.columns if c.startswith("WinProb (")]:
        fav = sim_preds.loc[sim_preds.groupby(["Year", "Round"])[col].idxmax()]
        rows[col.removeprefix("WinProb (").removesuffix(")")] = {
            "AvgWinChanceOfWinner%": float(100 * w[col].mean()),
            "FavouriteWon%": float(100 * (fav["FinishRank"] == 1).mean()),
            "WinnerLogLoss": float(-np.log(np.clip(w[col], 1e-4, 1)).mean())}
    return pd.DataFrame(rows).T.round(3)
