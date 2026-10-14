"""style_clusters pipeline: PCA + K-means, then SHAP per cluster."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    explain_clusters,
    fit_clusters,
    prepare_style_features,
    summarize_clusters,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                prepare_style_features,
                ["stats", "params:style_clusters"],
                "style_features",
                name="prepare_style_features",
            ),
            Node(
                fit_clusters,
                ["style_features", "params:style_clusters"],
                ["team_clusters", "cluster_info"],
                name="fit_clusters",
            ),
            Node(
                explain_clusters,
                ["stats", "team_clusters", "params:style_clusters"],
                ["cluster_top_features", "cluster_auc", "cluster_shap_plots"],
                name="explain_clusters",
            ),
            Node(
                summarize_clusters,
                ["cluster_top_features", "team_clusters", "stats"],
                "cluster_summary",
                name="summarize_clusters",
            ),
        ]
    )
