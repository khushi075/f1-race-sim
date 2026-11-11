"""Baseline race model: P(DNF) + finishing order among finishers.

Two parts, matching how the race simulator will work later:

1. DNF model        does the driver fail to take the chequered flag?
                    (retired, disqualified, withdrew/didn't start; lapped
                    cars that finish count as finishers)
2. Position model   finishing order among the drivers who finished

Each is compared against a simple benchmark:
    DNF:       the training-set DNF rate for everyone
    Position:  finish in grid order

Train on 2018-2024, test on 2025 (configurable). Every input uses only
information available before the race starts:
    Grid, PitLaneStart       starting slot (pit-lane start counts as last)
    QualiGapPct              this weekend's qualifying gap to teammate (%)
    DriverForm, TeamForm     mean finishing rank in their last 5 finishes
    RacePaceGapPct           mean race-pace gap to teammate, last 5 dry races
    DriverDNFRate, TeamDNFRate   share of DNFs in the last 10 races
    CircuitPosChange         mean |finish - grid| at this circuit in earlier
                             editions (how much the order usually shuffles)
    CircuitDNFRate           DNF rate at this circuit in earlier editions
"Recent" and circuit features look strictly at earlier races, so the 2025
test season never sees its own results.

Known limitations: team form carries over between seasons (a new car can be
much faster or slower), and form restarts when a team's id changes after a
rebrand (e.g. racing_point -> aston_martin).
"""
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FORM_WINDOW = 5
DNF_WINDOW = 10

POSITION_FEATURES = [
    "Grid", "PitLaneStart", "QualiGapPct", "DriverForm", "TeamForm",
    "RacePaceGapPct", "CircuitPosChange", "GridxCircuitPosChange",
]
DNF_FEATURES = ["Grid", "PitLaneStart", "DriverDNFRate", "TeamDNFRate", "CircuitDNFRate"]

_LAPPED = re.compile(r"^\+\d+ Laps?$")


def is_finished(status) -> bool:
    """Took the chequered flag: 'Finished', '+N Lap(s)' or 'Lapped'."""
    if not isinstance(status, str):
        return False
    return status in ("Finished", "Lapped") or bool(_LAPPED.match(status))


def _prior_mean(df: pd.DataFrame, key: str, value: str, window: int | None) -> pd.Series:
    """Per `key`, the mean of the last `window` non-missing `value`s strictly
    before each row (rows must be in race order). window=None: all earlier."""
    out = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby(key, sort=False).groups.items():
        s = df.loc[idx, value]
        vals = s.dropna()
        if vals.empty:
            continue
        roll = (vals.rolling(window, min_periods=1).mean() if window
                else vals.expanding().mean())
        out.loc[idx] = roll.reindex(s.index).ffill().shift(1).to_numpy()
    return out


def build_dataset(results: pd.DataFrame, quali_gaps: pd.DataFrame,
                  race_gaps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver per race, with targets and pre-race features.

    results:    load_results()                       (Jolpica)
    quali_gaps: features.teammate_gaps(load_quali())
    race_gaps:  race_pace.teammate_race_gaps(...)[0]
    """
    df = results.copy()
    df["Finished"] = df["Status"].map(is_finished)
    df["DNF"] = (~df["Finished"]).astype(int)
    df = df.sort_values(["Year", "Round", "Position"]).reset_index(drop=True)
    df["RaceIdx"] = df.groupby(["Year", "Round"], sort=True).ngroup()

    # Grid: pit-lane start (0) counts as starting last
    n_cars = df.groupby("RaceIdx")["DriverId"].transform("size")
    grid = pd.to_numeric(df["GridPosition"], errors="coerce").astype(float)
    df["PitLaneStart"] = (grid == 0).astype(int)
    df["Grid"] = grid.where(grid != 0, n_cars)

    # Target for the position model: finishing order among finishers only
    fin = df["Finished"]
    df["FinishRank"] = np.nan
    df.loc[fin, "FinishRank"] = df[fin].groupby("RaceIdx")["Position"].rank(method="first")
    df.loc[fin, "GridRankFinishers"] = df[fin].groupby("RaceIdx")["Grid"].rank(method="first")

    # Driver form and reliability (earlier races only)
    df["DriverForm"] = _prior_mean(df, "DriverId", "FinishRank", FORM_WINDOW)
    df["DriverDNFRate"] = _prior_mean(df, "DriverId", "DNF", DNF_WINDOW)

    # Team form and reliability: one value per team per race, then earlier races only
    team = (df.groupby(["TeamId", "RaceIdx"])
              .agg(TeamFinishRank=("FinishRank", "mean"), TeamDNF=("DNF", "mean"))
              .reset_index().sort_values("RaceIdx"))
    team["TeamForm"] = _prior_mean(team, "TeamId", "TeamFinishRank", FORM_WINDOW)
    team["TeamDNFRate"] = _prior_mean(team, "TeamId", "TeamDNF", DNF_WINDOW)
    df = df.merge(team[["TeamId", "RaceIdx", "TeamForm", "TeamDNFRate"]],
                  on=["TeamId", "RaceIdx"], how="left")

    # Circuit history: earlier editions of the same event; if none, the mean
    # of all earlier races
    df["PosChange"] = (df["FinishRank"] - df["GridRankFinishers"]).abs()
    race = (df.groupby(["RaceIdx", "Event"])
              .agg(RacePosChange=("PosChange", "mean"), RaceDNF=("DNF", "mean"))
              .reset_index().sort_values("RaceIdx"))
    race["_all"] = 0
    for col, src in (("CircuitPosChange", "RacePosChange"), ("CircuitDNFRate", "RaceDNF")):
        by_event = _prior_mean(race, "Event", src, None)
        overall = _prior_mean(race, "_all", src, None)
        race[col] = by_event.fillna(overall)
    df = df.merge(race[["RaceIdx", "CircuitPosChange", "CircuitDNFRate"]], on="RaceIdx", how="left")
    df["GridxCircuitPosChange"] = df["Grid"] * df["CircuitPosChange"]

    # This weekend's qualifying gap (outliers left missing, as in feature 1)
    q = quali_gaps.loc[~quali_gaps["Outlier"], ["Year", "Round", "Driver", "GapPct"]]
    q = q.rename(columns={"Driver": "Abbreviation", "GapPct": "QualiGapPct"})
    df = df.merge(q, on=["Year", "Round", "Abbreviation"], how="left")

    # Race-pace gap from earlier dry races
    rp = race_gaps[["Year", "Round", "Driver", "GapPct"]].rename(
        columns={"Driver": "Abbreviation", "GapPct": "RacePaceGap"})
    df = df.merge(rp, on=["Year", "Round", "Abbreviation"], how="left")
    df = df.sort_values(["RaceIdx", "Position"]).reset_index(drop=True)
    df["RacePaceGapPct"] = _prior_mean(df, "DriverId", "RacePaceGap", FORM_WINDOW)
    return df


def _rank_within_race(df: pd.DataFrame, score: pd.Series) -> pd.Series:
    return score.groupby(df["RaceIdx"]).rank(method="first")


def _position_scores(test: pd.DataFrame, pred_rank: pd.Series) -> dict:
    per_race = [spearmanr(g["FinishRank"], pred_rank.loc[g.index]).statistic
                for _, g in test.groupby("RaceIdx")]
    err = (pred_rank - test["FinishRank"]).abs().dropna()  # e.g. grid unknown
    return {"MeanSpearman": float(np.nanmean(per_race)),
            "MAE_positions": float(err.mean()),
            "ExactPct": float(100 * (err == 0).mean()),
            "DriversScored": int(len(err))}


def fit_evaluate(df: pd.DataFrame, test_year: int = 2025, seed: int = 0) -> dict:
    """Fit both models on seasons before test_year; score them on test_year."""
    train, test = df[df["Year"] < test_year], df[df["Year"] == test_year]
    out = {"n_train_rows": len(train), "n_test_rows": len(test)}

    # --- DNF model -------------------------------------------------------
    Xtr, ytr = train[DNF_FEATURES], train["DNF"]
    Xte, yte = test[DNF_FEATURES], test["DNF"]
    dnf_models = {
        "Linear (logistic)": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                           StandardScaler(), LogisticRegression(max_iter=1000)),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
            random_state=seed),
    }
    base_rate = float(ytr.mean())
    preds = {"Base rate": np.full(len(yte), base_rate)}
    for name, m in dnf_models.items():
        m.fit(Xtr, ytr)
        preds[name] = m.predict_proba(Xte)[:, 1]
    out["dnf_metrics"] = pd.DataFrame([
        {"Model": name, "LogLoss": log_loss(yte, p, labels=[0, 1]),
         "Brier": brier_score_loss(yte, p),
         "AUC": roc_auc_score(yte, p) if name != "Base rate" else 0.5}
        for name, p in preds.items()
    ]).set_index("Model").round(4)
    out["dnf_models"] = dnf_models
    out["dnf_test_pred"] = test[["Year", "Round", "Event", "Abbreviation", "TeamName", "Grid", "DNF", "Status"]].assign(
        **{f"P_DNF ({k})": v for k, v in preds.items()})

    # --- Position model (finishers only) --------------------------------
    trf, tef = train[train["Finished"]], test[test["Finished"]]
    Xtr, ytr = trf[POSITION_FEATURES], trf["FinishRank"]
    Xte = tef[POSITION_FEATURES]
    pos_models = {
        "Linear (ridge)": make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                        StandardScaler(), Ridge(alpha=1.0)),
        "Gradient boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
            random_state=seed),
    }
    ranks = {"Grid order": tef["GridRankFinishers"]}
    for name, m in pos_models.items():
        m.fit(Xtr, ytr)
        ranks[name] = _rank_within_race(tef, pd.Series(m.predict(Xte), index=tef.index))
    # Score every model on the same drivers (all predictions available)
    common = pd.concat(ranks, axis=1).notna().all(axis=1)
    scored = tef[common]
    out["position_metrics"] = pd.DataFrame(
        [{"Model": k, **_position_scores(scored, v[common])} for k, v in ranks.items()]
    ).set_index("Model").round(3)
    out["position_unscored"] = tef.loc[~common, ["Round", "Event", "Abbreviation", "Grid"]]
    out["position_models"] = pos_models
    out["position_test_pred"] = tef[["Year", "Round", "Event", "Abbreviation", "TeamName",
                                     "Grid", "FinishRank"]].assign(
        **{f"Pred ({k})": v.astype("Int64") for k, v in ranks.items()})

    # Per-race Spearman, for charts
    rows = []
    for race_idx, g in tef.groupby("RaceIdx"):
        row = {"Round": int(g["Round"].iloc[0]), "Event": g["Event"].iloc[0]}
        for k, v in ranks.items():
            row[k] = spearmanr(g["FinishRank"], v.loc[g.index]).statistic
        rows.append(row)
    out["position_per_race"] = pd.DataFrame(rows)

    # --- What drives the predictions ------------------------------------
    ridge = pos_models["Linear (ridge)"]
    names = ridge[:-1].get_feature_names_out(POSITION_FEATURES)
    out["position_linear_coefs"] = (pd.Series(ridge[-1].coef_, index=names)
                                    .sort_values(key=np.abs, ascending=False).round(3))
    imp = permutation_importance(pos_models["Gradient boosting"], Xte, tef["FinishRank"],
                                 scoring="neg_mean_absolute_error", n_repeats=10, random_state=seed)
    out["position_gb_importance"] = (pd.Series(imp.importances_mean, index=POSITION_FEATURES)
                                     .sort_values(ascending=False).round(3))
    return out
