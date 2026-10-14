"""Nodes for the drivers pipeline: feature prep, model training, SHAP."""

import logging
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

logger = logging.getLogger(__name__)

ID_COLS = ["team", "season"]
TARGET_PREFIX = "target__"


def prepare_features(stats: pd.DataFrame, parameters: dict[str, Any]) -> pd.DataFrame:
    """Every stat becomes a candidate feature; raw targets are kept alongside.

    Optionally z-scores every stat within its season so that league-wide
    changes in how stats are recorded do not dominate the model. Targets are
    stored raw in ``target__<column>`` so SHAP stays in real units.
    """
    stat_cols = [c for c in stats.columns if c not in ID_COLS]
    df = stats[ID_COLS + stat_cols].copy()
    if parameters["normalize_by_season"]:
        grouped = df.groupby("season")[stat_cols]
        df[stat_cols] = (df[stat_cols] - grouped.transform("mean")) / grouped.transform("std")
    for column in parameters["targets"].values():
        df[f"{TARGET_PREFIX}{column}"] = stats[column]
    return df


def _select_features(
    model_input: pd.DataFrame, target: str, parameters: dict[str, Any]
) -> pd.DataFrame:
    drop = (
        set(parameters["exclude_always"])
        | set(parameters["exclude_by_target"][target])
        | {target}
    )
    cols = [
        c
        for c in model_input.columns
        if c not in ID_COLS and not c.startswith(TARGET_PREFIX) and c not in drop
    ]
    return model_input[cols]


def train_model(
    model_input: pd.DataFrame, target: str, parameters: dict[str, Any]
) -> tuple[lgb.LGBMRegressor, dict[str, float]]:
    """Fit a LightGBM regressor and report season-grouped CV R²."""
    X = _select_features(model_input, target, parameters)
    y = model_input[f"{TARGET_PREFIX}{target}"]
    model_params = parameters["model"]
    cv = GroupKFold(n_splits=parameters["cv_splits"])
    scores = []
    for train_idx, test_idx in cv.split(X, y, groups=model_input["season"]):
        m = lgb.LGBMRegressor(**model_params, verbose=-1)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(r2_score(y.iloc[test_idx], m.predict(X.iloc[test_idx])))
    metrics = {"cv_r2_mean": float(np.mean(scores)), "cv_r2_std": float(np.std(scores))}
    logger.info("%s: CV R2 %.3f +/- %.3f", target, metrics["cv_r2_mean"], metrics["cv_r2_std"])
    model = lgb.LGBMRegressor(**model_params, verbose=-1).fit(X, y)
    return model, metrics


def compute_shap(
    model: lgb.LGBMRegressor,
    model_input: pd.DataFrame,
    target: str,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    """SHAP values per team-season, one column per feature."""
    X = _select_features(model_input, target, parameters)
    values = shap.TreeExplainer(model).shap_values(X)
    out = pd.DataFrame(values, columns=X.columns, index=model_input.index)
    return pd.concat([model_input[ID_COLS], out], axis=1)


def rank_drivers(shap_values: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP per feature, largest first."""
    imp = shap_values.drop(columns=ID_COLS).abs().mean().sort_values(ascending=False)
    return imp.rename("mean_abs_shap").rename_axis("feature").reset_index()


def plot_shap(
    shap_values: pd.DataFrame, model_input: pd.DataFrame, target: str, parameters: dict[str, Any]
) -> dict[str, plt.Figure]:
    """Beeswarm and bar summary plots."""
    X = _select_features(model_input, target, parameters)
    sv = shap_values.drop(columns=ID_COLS).to_numpy()
    top = parameters["top_n_plot"]
    figs = {}
    for kind, name in [("dot", "beeswarm.png"), ("bar", "importance.png")]:
        plt.figure()
        shap.summary_plot(sv, X, plot_type=kind, max_display=top, show=False)
        plt.title(f"What drives {target}")
        plt.tight_layout()
        figs[name] = plt.gcf()
    return figs
