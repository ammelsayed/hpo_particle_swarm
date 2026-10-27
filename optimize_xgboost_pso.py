#!/usr/bin/env python3
"""
XGBoost hyperparameter optimization via Particle Swarm Optimization (PSO).

Uses the mealpy library's PSO.OriginalPSO (classic Kennedy & Eberhart 1995
algorithm) — we do NOT implement PSO ourselves. The PSO optimizer searches
over continuous hyperparameter bounds; each particle is evaluated by training
XGBoost on GPU and scoring hold-out ROC-AUC.

Package: mealpy (https://github.com/thieu1995/mealpy)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from mealpy import FloatVar, PSO
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_SAMPLES = 20_000
N_FEATURES = 60
N_INFORMATIVE = 40
TEST_SIZE = 0.2
N_ESTIMATORS = 200

# PSO settings (kept moderate so a full run finishes in a few minutes on GPU)
PSO_EPOCH = 15          # iterations
PSO_POP_SIZE = 12       # particles
PSO_C1 = 2.05
PSO_C2 = 2.05
PSO_W = 0.4

# Hyperparameter search space (order must match decode_params)
#   learning_rate, max_depth, subsample, colsample_bytree,
#   min_child_weight, reg_lambda
LB = [0.01, 2.0, 0.5, 0.5, 1.0, 0.1]
UB = [0.3, 10.0, 1.0, 1.0, 10.0, 10.0]
PARAM_NAMES = [
    "learning_rate",
    "max_depth",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_lambda",
]

OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results.json"


def decode_params(solution: np.ndarray) -> dict:
    """Map a continuous PSO vector to XGBoost kwargs."""
    return {
        "learning_rate": float(solution[0]),
        "max_depth": int(round(solution[1])),
        "subsample": float(solution[2]),
        "colsample_bytree": float(solution[3]),
        "min_child_weight": float(solution[4]),
        "reg_lambda": float(solution[5]),
        "n_estimators": N_ESTIMATORS,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
        "verbosity": 0,
    }


def make_data():
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=10,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.02,
        random_state=RANDOM_SEED,
    )
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )


def check_gpu() -> str:
    """Train a tiny model on CUDA to confirm GPU works; return a status string."""
    import warnings
    X = np.random.RandomState(0).rand(500, 8).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    info = xgb.build_info() if hasattr(xgb, "build_info") else {}
    cuda_ver = info.get("CUDA_VERSION", "unknown")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            booster = xgb.train(
                {
                    "objective": "binary:logistic",
                    "tree_method": "hist",
                    "device": "cuda",
                    "max_depth": 2,
                    "verbosity": 0,
                },
                dtrain,
                num_boost_round=5,
            )
        msgs = " | ".join(str(w.message) for w in caught)
        if "No visible GPU" in msgs or "Device is changed from GPU to CPU" in msgs:
            raise RuntimeError(
                "XGBoost fell back to CPU. On this host install the CUDA-12 wheel: "
                "pip uninstall -y xgboost xgboost-cu12 && pip install xgboost-cu12. "
                f"build CUDA_VERSION={cuda_ver}. Warnings: {msgs}"
            )
        return (
            f"cuda OK (boosted_rounds={booster.num_boosted_rounds()}, "
            f"CUDA_VERSION={cuda_ver})"
        )
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"XGBoost GPU training failed: {exc}") from exc


def main() -> None:
    print("=" * 60)
    print("PSO (mealpy) + XGBoost GPU hyperparameter optimization")
    print("=" * 60)
    print(f"mealpy version : {__import__('mealpy').__version__}")
    print(f"xgboost version: {xgb.__version__}")
    print(f"GPU check      : {check_gpu()}")

    X_train, X_test, y_train, y_test = make_data()
    print(
        f"Data           : make_classification "
        f"train={X_train.shape} test={X_test.shape}"
    )

    eval_count = {"n": 0}

    def objective(solution: np.ndarray) -> float:
        """Minimize 1 - AUC (mealpy minmax='min')."""
        params = decode_params(solution)
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
        eval_count["n"] += 1
        # Print occasional progress
        if eval_count["n"] % PSO_POP_SIZE == 0 or eval_count["n"] <= 3:
            print(
                f"  eval #{eval_count['n']:3d}  AUC={auc:.5f}  "
                f"lr={params['learning_rate']:.4f} "
                f"depth={params['max_depth']} "
                f"sub={params['subsample']:.3f}"
            )
        return 1.0 - auc

    problem = {
        "bounds": FloatVar(lb=LB, ub=UB, name="xgb_hparams"),
        "obj_func": objective,
        "minmax": "min",
        "log_to": "console",
        "name": "xgboost_pso",
    }

    optimizer = PSO.OriginalPSO(
        epoch=PSO_EPOCH,
        pop_size=PSO_POP_SIZE,
        c1=PSO_C1,
        c2=PSO_C2,
        w=PSO_W,
    )

    t0 = time.perf_counter()
    g_best = optimizer.solve(problem)
    elapsed = time.perf_counter() - t0

    best_params = decode_params(g_best.solution)
    best_auc = 1.0 - float(g_best.target.fitness)

    # Final retrain + report
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    final_auc = float(roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1]))

    results = {
        "package": "mealpy",
        "package_version": __import__("mealpy").__version__,
        "algorithm": "PSO.OriginalPSO (Kennedy & Eberhart 1995)",
        "xgboost_version": xgb.__version__,
        "device": "cuda",
        "data": {
            "generator": "sklearn.datasets.make_classification",
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "n_informative": N_INFORMATIVE,
            "test_size": TEST_SIZE,
            "random_seed": RANDOM_SEED,
        },
        "pso": {
            "epoch": PSO_EPOCH,
            "pop_size": PSO_POP_SIZE,
            "c1": PSO_C1,
            "c2": PSO_C2,
            "w": PSO_W,
            "evaluations": eval_count["n"],
        },
        "search_space": {
            name: {"lb": lb, "ub": ub}
            for name, lb, ub in zip(PARAM_NAMES, LB, UB)
        },
        "best_params": {
            k: (int(v) if k == "max_depth" else float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in best_params.items()
            if k in PARAM_NAMES or k == "n_estimators"
        },
        "best_auc_during_search": best_auc,
        "best_auc_retrain": final_auc,
        "runtime_seconds": round(elapsed, 2),
    }

    # Make JSON-safe
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    results = _sanitize(results)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print("DONE")
    print(f"Best AUC (search) : {best_auc:.6f}")
    print(f"Best AUC (retrain): {final_auc:.6f}")
    print(f"Best params       : {json.dumps(results['best_params'], indent=2)}")
    print(f"Runtime           : {elapsed:.1f}s")
    print(f"Results written   : {RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
