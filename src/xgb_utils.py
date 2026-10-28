"""XGBoost training helpers for strategy A (fixed n_estimators) and B (early stopping)."""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb

from . import config as cfg


def detect_device() -> str:
    """Return 'cuda' if a quick GPU train works without CPU fallback, else 'cpu'."""
    import warnings

    X = np.random.RandomState(0).rand(200, 4).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            xgb.train(
                {
                    "objective": "binary:logistic",
                    "tree_method": cfg.XGB_TREE_METHOD,
                    "device": "cuda",
                    "max_depth": 2,
                    "verbosity": 0,
                },
                dtrain,
                num_boost_round=2,
            )
        msgs = " | ".join(str(w.message) for w in caught)
        if "No visible GPU" in msgs or "Device is changed from GPU to CPU" in msgs:
            return "cpu"
        return "cuda"
    except Exception:
        return "cpu"


DEVICE = detect_device()


def decode_core_params(vec: np.ndarray | list[float]) -> dict[str, Any]:
    """Map continuous vector -> core XGBoost hyperparameters (no n_estimators)."""
    return {
        "learning_rate": float(vec[0]),
        "max_depth": int(round(float(vec[1]))),
        "subsample": float(vec[2]),
        "colsample_bytree": float(vec[3]),
        "min_child_weight": float(vec[4]),
        "reg_lambda": float(vec[5]),
    }


def default_core_params() -> dict[str, Any]:
    """XGBoost defaults used for the no-HPO baseline."""
    return {
        "learning_rate": 0.3,
        "max_depth": 6,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
    }


def _base_kwargs(core: dict[str, Any]) -> dict[str, Any]:
    return {
        **core,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": cfg.XGB_TREE_METHOD,
        "device": DEVICE,
        "random_state": cfg.RANDOM_SEED,
        "n_jobs": 1,
        "verbosity": 0,
    }


def train_strategy_a(
    X_train,
    y_train,
    core_params: dict[str, Any],
    n_estimators: int,
):
    """Strategy A: train for exactly n_estimators rounds (no early stopping)."""
    model = xgb.XGBClassifier(
        **_base_kwargs(core_params),
        n_estimators=int(n_estimators),
        early_stopping_rounds=None,
    )
    model.fit(X_train, y_train)
    return model


def train_strategy_b(
    X_train,
    y_train,
    X_val,
    y_val,
    core_params: dict[str, Any],
    n_estimators_cap: int = cfg.N_EST_B_CAP,
    early_stopping_rounds: int = cfg.EARLY_STOPPING_ROUNDS,
):
    """Strategy B: high n_estimators + early stopping on validation AUC."""
    model = xgb.XGBClassifier(
        **_base_kwargs(core_params),
        n_estimators=int(n_estimators_cap),
        early_stopping_rounds=int(early_stopping_rounds),
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def predict_scores(model, X) -> np.ndarray:
    """BDT score = predicted probability of the positive class (class 1)."""
    return model.predict_proba(X)[:, 1]


def best_n_estimators(model) -> int:
    """Effective number of trees used (best_iteration+1 when early stopping)."""
    bi = getattr(model, "best_iteration", None)
    if bi is not None:
        return int(bi) + 1
    return int(model.n_estimators)


def val_auc_for_hpo(model, X_val, y_val) -> float:
    """Validation AUC — the only metric used inside HPO loops."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_val, predict_scores(model, X_val)))
