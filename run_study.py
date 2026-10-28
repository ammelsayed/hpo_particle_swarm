#!/usr/bin/env python3
"""Run the full HPO comparison study and write results + figures.

Usage:
    python run_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src.data import make_splits
from src.evaluate import (
    evaluate_on_test,
    plot_bdt_histograms,
    plot_metric_comparison,
    summarize_findings,
)
from src.hpo_methods import run_all_hpo
from src.xgb_utils import DEVICE


def main() -> None:
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"XGBoost device: {DEVICE}")
    print(
        f"Data: make_classification n={cfg.N_SAMPLES}, "
        f"features={cfg.N_FEATURES}, split={cfg.TRAIN_FRAC}/{cfg.VAL_FRAC}/{cfg.TEST_FRAC}"
    )

    data = make_splits()
    print(
        f"Shapes: train={data.X_train.shape}, val={data.X_val.shape}, test={data.X_test.shape}"
    )

    hpo_rows = run_all_hpo(data)
    (cfg.RESULTS_DIR / "hpo_search.json").write_text(json.dumps(hpo_rows, indent=2))

    df, score_store = evaluate_on_test(data, hpo_rows)
    df.to_csv(cfg.RESULTS_DIR / "test_metrics.csv", index=False)
    # Persist test scores for reproducibility of plots
    score_path = cfg.RESULTS_DIR / "test_scores.npz"
    np_payload = {"y_test": data.y_test}
    np_payload.update(score_store)
    import numpy as np

    np.savez_compressed(score_path, **np_payload)

    fig_paths = []
    fig_paths += plot_bdt_histograms(data.y_test, score_store, cfg.FIGURES_DIR)
    fig_paths += plot_metric_comparison(df, cfg.FIGURES_DIR)

    summary = summarize_findings(df)
    summary["device"] = DEVICE
    summary["figures"] = [str(p.relative_to(ROOT)) for p in fig_paths]
    (cfg.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Test metrics ===")
    cols = [
        "method",
        "strategy",
        "n_estimators",
        "test_auc",
        "test_asimov_z",
        "auc_gain",
        "asimov_gain",
    ]
    print(df[cols].to_string(index=False))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote results to {cfg.RESULTS_DIR}")
    print(f"Wrote figures to {cfg.FIGURES_DIR}")


if __name__ == "__main__":
    main()
