"""Drivers pipeline: one model + SHAP explanation per target."""

from kedro.pipeline import Node, Pipeline

from .nodes import compute_shap, plot_shap, prepare_features, rank_drivers, train_model


def create_pipeline(**kwargs) -> Pipeline:
    nodes = [
        Node(
            prepare_features,
            ["stats", "params:drivers"],
            "model_input",
            name="prepare_features",
        )
    ]
    for key in ("scored", "conceded", "on_target"):
        target = f"params:drivers.targets.{key}"
        nodes += [
            Node(
                train_model,
                ["model_input", target, "params:drivers"],
                [f"model_{key}", f"metrics_{key}"],
                name=f"train_{key}",
            ),
            Node(
                compute_shap,
                [f"model_{key}", "model_input", target, "params:drivers"],
                f"shap_values_{key}",
                name=f"shap_{key}",
            ),
            Node(
                rank_drivers,
                f"shap_values_{key}",
                f"drivers_{key}",
                name=f"rank_{key}",
            ),
            Node(
                plot_shap,
                [f"shap_values_{key}", "model_input", target, "params:drivers"],
                f"shap_plots_{key}",
                name=f"plot_{key}",
            ),
        ]
    return Pipeline(nodes)
