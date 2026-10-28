"""Final test-set evaluation and figure generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg
from .data import DatasetSplits
from .metrics import binned_asimov_significance, roc_auc
from .xgb_utils import (
    predict_scores,
    train_strategy_a,
)


METHOD_LABELS = {
    "baseline": "Baseline (no HPO)",
    "optuna": "Optuna (TPE)",
    "mealpy": "mealpy PSO",
    "pyswarms": "pyswarms PSO",
}

STRATEGY_LABELS = {
    "A_optimize_n_estimators": "A: optimize n_estimators",
    "B_early_stopping": "B: early stopping",
}


def train_final_model(data: DatasetSplits, row: dict[str, Any]):
    """Retrain final model from HPO result.

    Strategy A: train for the searched n_estimators (no early stopping).
    Strategy B: train for the best_iteration discovered during HPO (fixed
    n_estimators, no further early stopping). Validation is not used here;
    it was only used inside HPO for early stopping / fitness.
    """
    core = row["core_params"]
    return train_strategy_a(data.X_train, data.y_train, core, int(row["n_estimators"]))


def evaluate_on_test(data: DatasetSplits, hpo_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Train finals and score ONLY on the test set."""
    records = []
    score_store: dict[str, np.ndarray] = {}

    for row in hpo_rows:
        key = f"{row['method']}__{row['strategy']}"
        model = train_final_model(data, row)
        scores = predict_scores(model, data.X_test)
        score_store[key] = scores
        auc = roc_auc(data.y_test, scores)
        z = binned_asimov_significance(
            data.y_test,
            scores,
            n_bins=cfg.ASIMOV_N_BINS,
            b_floor=cfg.ASIMOV_B_FLOOR,
        )
        records.append(
            {
                "method": row["method"],
                "strategy": row["strategy"],
                "n_estimators": row["n_estimators"],
                "val_auc": row["best_val_auc"],
                "test_auc": auc,
                "test_asimov_z": z,
                "hpo_runtime_s": row["runtime_seconds"],
                "core_params": json.dumps(row["core_params"]),
            }
        )

    df = pd.DataFrame(records)

    # Gains vs matching baseline strategy
    for strat in df["strategy"].unique():
        base = df[(df["method"] == "baseline") & (df["strategy"] == strat)]
        if base.empty:
            continue
        base_auc = float(base["test_auc"].iloc[0])
        base_z = float(base["test_asimov_z"].iloc[0])
        mask = df["strategy"] == strat
        df.loc[mask, "auc_gain"] = df.loc[mask, "test_auc"] - base_auc
        df.loc[mask, "asimov_gain"] = df.loc[mask, "test_asimov_z"] - base_z

    return df, score_store


def plot_bdt_histograms(
    y_test: np.ndarray,
    score_store: dict[str, np.ndarray],
    out_dir: Path,
) -> list[Path]:
    """One figure per method (both strategies as panels) + a 2x2 overview for strategy B."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    methods = ["baseline", "optuna", "mealpy", "pyswarms"]
    strategies = ["A_optimize_n_estimators", "B_early_stopping"]

    # Per-method two-panel figure
    for method in methods:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax, strat in zip(axes, strategies):
            key = f"{method}__{strat}"
            if key not in score_store:
                ax.set_visible(False)
                continue
            scores = score_store[key]
            sig = scores[y_test == 1]
            bkg = scores[y_test == 0]
            bins = np.linspace(0, 1, 31)
            ax.hist(bkg, bins=bins, histtype="step", linewidth=1.6, label="Background (y=0)", density=True)
            ax.hist(sig, bins=bins, histtype="step", linewidth=1.6, label="Signal (y=1)", density=True)
            ax.set_xlabel("BDT score (P(y=1))")
            ax.set_ylabel("Density")
            ax.set_title(STRATEGY_LABELS[strat])
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"BDT score distributions — {METHOD_LABELS[method]}", fontsize=12)
        fig.tight_layout()
        path = out_dir / f"bdt_scores_{method}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # Overview: strategy B only (early stopping), all methods
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    for ax, method in zip(axes.ravel(), methods):
        key = f"{method}__B_early_stopping"
        scores = score_store[key]
        sig = scores[y_test == 1]
        bkg = scores[y_test == 0]
        bins = np.linspace(0, 1, 31)
        ax.hist(bkg, bins=bins, histtype="step", linewidth=1.6, label="Background (y=0)", density=True)
        ax.hist(sig, bins=bins, histtype="step", linewidth=1.6, label="Signal (y=1)", density=True)
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("BDT score (P(y=1))")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("BDT scores on test set — strategy B (early stopping)", fontsize=13)
    fig.tight_layout()
    path = out_dir / "bdt_scores_overview_B.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


def plot_metric_comparison(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    methods = ["baseline", "optuna", "mealpy", "pyswarms"]
    method_short = [METHOD_LABELS[m] for m in methods]
    x = np.arange(len(methods))
    width = 0.35

    def _vals(metric: str, strategy: str) -> list[float]:
        out = []
        for m in methods:
            row = df[(df["method"] == m) & (df["strategy"] == strategy)]
            out.append(float(row[metric].iloc[0]) if len(row) else np.nan)
        return out

    # AUC comparison
    fig, ax = plt.subplots(figsize=(9, 4.5))
    a_vals = _vals("test_auc", "A_optimize_n_estimators")
    b_vals = _vals("test_auc", "B_early_stopping")
    ax.bar(x - width / 2, a_vals, width, label="A: optimize n_estimators")
    ax.bar(x + width / 2, b_vals, width, label="B: early stopping")
    ax.set_xticks(x)
    ax.set_xticklabels(method_short, rotation=15, ha="right")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Test AUC by method and n_estimators strategy")
    ax.legend(frameon=False)
    ymin = min(a_vals + b_vals) - 0.01
    ax.set_ylim(max(0.8, ymin), 1.0)
    fig.tight_layout()
    path = out_dir / "compare_auc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # Asimov Z comparison
    fig, ax = plt.subplots(figsize=(9, 4.5))
    a_vals = _vals("test_asimov_z", "A_optimize_n_estimators")
    b_vals = _vals("test_asimov_z", "B_early_stopping")
    ax.bar(x - width / 2, a_vals, width, label="A: optimize n_estimators")
    ax.bar(x + width / 2, b_vals, width, label="B: early stopping")
    ax.set_xticks(x)
    ax.set_xticklabels(method_short, rotation=15, ha="right")
    ax.set_ylabel("Binned Asimov significance Z")
    ax.set_title("Test binned Asimov Z by method and n_estimators strategy")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "compare_asimov.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # Gains vs baseline (HPO methods only)
    hpo_methods = ["optuna", "mealpy", "pyswarms"]
    hx = np.arange(len(hpo_methods))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, title in zip(
        axes,
        ["auc_gain", "asimov_gain"],
        ["AUC gain vs baseline", "Asimov Z gain vs baseline"],
    ):
        a_vals = []
        b_vals = []
        for m in hpo_methods:
            for strat, bucket in [
                ("A_optimize_n_estimators", a_vals),
                ("B_early_stopping", b_vals),
            ]:
                row = df[(df["method"] == m) & (df["strategy"] == strat)]
                bucket.append(float(row[metric].iloc[0]) if len(row) else np.nan)
        ax.bar(hx - width / 2, a_vals, width, label="A: optimize n_estimators")
        ax.bar(hx + width / 2, b_vals, width, label="B: early stopping")
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_xticks(hx)
        ax.set_xticklabels([METHOD_LABELS[m] for m in hpo_methods], rotation=15, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "compare_gains.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


def summarize_findings(df: pd.DataFrame) -> dict[str, Any]:
    """Pick winners for README / report."""
    best_auc_row = df.loc[df["test_auc"].idxmax()]
    best_z_row = df.loc[df["test_asimov_z"].idxmax()]

    per_method_ab = {}
    for method in ["optuna", "mealpy", "pyswarms", "baseline"]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        a = sub[sub["strategy"] == "A_optimize_n_estimators"].iloc[0]
        b = sub[sub["strategy"] == "B_early_stopping"].iloc[0]
        # Prefer B if AUC >= A (tie -> early stopping as preferred default)
        winner = "B_early_stopping" if b["test_auc"] >= a["test_auc"] else "A_optimize_n_estimators"
        per_method_ab[method] = {
            "A_test_auc": float(a["test_auc"]),
            "B_test_auc": float(b["test_auc"]),
            "A_test_asimov_z": float(a["test_asimov_z"]),
            "B_test_asimov_z": float(b["test_asimov_z"]),
            "winner_by_auc": winner,
            "winner_by_asimov": (
                "B_early_stopping"
                if b["test_asimov_z"] >= a["test_asimov_z"]
                else "A_optimize_n_estimators"
            ),
        }

    return {
        "best_by_auc": {
            "method": best_auc_row["method"],
            "strategy": best_auc_row["strategy"],
            "test_auc": float(best_auc_row["test_auc"]),
            "test_asimov_z": float(best_auc_row["test_asimov_z"]),
        },
        "best_by_asimov": {
            "method": best_z_row["method"],
            "strategy": best_z_row["strategy"],
            "test_auc": float(best_z_row["test_auc"]),
            "test_asimov_z": float(best_z_row["test_asimov_z"]),
        },
        "per_method_A_vs_B": per_method_ab,
    }
