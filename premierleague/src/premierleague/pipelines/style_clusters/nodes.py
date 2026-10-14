"""Nodes for the style_clusters pipeline.

PCA -> K-means -> one binary target per cluster -> XGBoost per cluster -> SHAP.
"""

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_auc_score, silhouette_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

logger = logging.getLogger(__name__)

ID_COLS = ["team", "season"]


def _style_columns(stats: pd.DataFrame, parameters: dict[str, Any]) -> list[str]:
    outcomes = set(parameters["outcome_cols"])
    return [c for c in stats.columns if c not in ID_COLS and c not in outcomes]


def prepare_style_features(stats: pd.DataFrame, parameters: dict[str, Any]) -> pd.DataFrame:
    """Season z-scored playing-style stats; outcomes (goals, wins...) are left out."""
    cols = _style_columns(stats, parameters)
    grouped = stats.groupby("season")[cols]
    z = (stats[cols] - grouped.transform("mean")) / grouped.transform("std")
    return pd.concat([stats[ID_COLS], z], axis=1)


def fit_clusters(
    style_features: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """PCA then K-means. k is chosen by silhouette over ``k_range``."""
    cols = [c for c in style_features.columns if c not in ID_COLS]
    usable = [c for c in cols if style_features[c].isna().mean() <= parameters["max_missing"]]
    X = style_features[usable].fillna(0.0)  # z-scored, so 0 is the season average
    pca = PCA(n_components=parameters["pca_variance"], random_state=parameters["seed"])
    pcs = pca.fit_transform(X)
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(*parameters["k_range"]):
        labels = KMeans(n_clusters=k, n_init=20, random_state=parameters["seed"]).fit_predict(pcs)
        score = silhouette_score(pcs, labels)
        logger.info("k=%d silhouette=%.3f", k, score)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels
    if parameters.get("k") is not None:
        best_k = parameters["k"]
        best_labels = KMeans(
            n_clusters=best_k, n_init=20, random_state=parameters["seed"]
        ).fit_predict(pcs)
        best_score = silhouette_score(pcs, best_labels)
    info = {
        "k": int(best_k),
        "silhouette": float(best_score),
        "n_components": int(pca.n_components_),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "clustered_on": usable,
    }
    logger.info("Chose k=%d (silhouette %.3f, %d PCs)", best_k, best_score, pca.n_components_)
    clusters = style_features[ID_COLS].copy()
    clusters["cluster"] = best_labels
    return clusters, info


def explain_clusters(
    stats: pd.DataFrame, clusters: pd.DataFrame, parameters: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, float], dict[str, plt.Figure]]:
    """Per cluster: invert the target, fit XGBoost, SHAP, keep the top features."""
    cols = _style_columns(stats, parameters)
    X = stats[cols]
    top_n = parameters["top_n"]
    rows, aucs, figs = [], {}, {}
    for cluster in sorted(clusters["cluster"].unique()):
        y = (clusters["cluster"] == cluster).astype(int)  # TGT_cluster_X
        model_params = dict(parameters["xgb"], random_state=parameters["seed"])
        # honest signal check: out-of-fold AUC
        n_splits = int(min(5, y.sum()))
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=parameters["seed"]).split(X, y):
            oof[te] = xgb.XGBClassifier(**model_params).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
        aucs[f"cluster_{cluster}"] = float(roc_auc_score(y, oof))
        model = xgb.XGBClassifier(**model_params).fit(X, y)
        sv = shap.TreeExplainer(model).shap_values(X)
        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
        for rank, feat in enumerate(mean_abs.index[:top_n], start=1):
            valid = X[feat].notna()
            corr = np.corrcoef(X.loc[valid, feat], sv[valid.to_numpy(), cols.index(feat)])[0, 1]
            rows.append(
                {
                    "cluster": int(cluster),
                    "rank": rank,
                    "feature": feat,
                    "mean_abs_shap": float(mean_abs[feat]),
                    "direction": "high pushes toward" if corr > 0 else "high pushes against",
                    "corr_feature_shap": float(corr),
                }
            )
        plt.figure()
        shap.summary_plot(sv, X, max_display=top_n, show=False)
        plt.title(f"Cluster {cluster}: what defines it")
        plt.tight_layout()
        figs[f"cluster_{cluster}.png"] = plt.gcf()
    return pd.DataFrame(rows), aucs, figs


def summarize_clusters(
    top_features: pd.DataFrame, clusters: pd.DataFrame, stats: pd.DataFrame
) -> pd.DataFrame:
    """cluster | size | goals | conceded | heaviest feature | SHAP | direction."""
    merged = clusters.merge(stats, on=ID_COLS)
    prof = merged.groupby("cluster").agg(
        n=("team", "size"),
        goals=("goals", "mean"),
        goals_conceded=("goals_conceded", "mean"),
        wins=("wins", "mean"),
        examples=("team", lambda s: ", ".join(s.value_counts().index[:3])),
    )
    top1 = top_features[top_features["rank"] == 1].set_index("cluster")
    out = prof.join(
        top1[["feature", "mean_abs_shap", "direction"]].rename(
            columns={"feature": "heaviest_feature"}
        )
    )
    return out.round(2).sort_values("goals", ascending=False).reset_index()
