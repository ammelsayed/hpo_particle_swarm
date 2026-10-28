"""HPO runners: baseline, Optuna, mealpy PSO, pyswarms PSO (strategies A and B)."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from . import config as cfg
from .data import DatasetSplits
from .xgb_utils import (
    decode_core_params,
    default_core_params,
    train_strategy_a,
    train_strategy_b,
    val_auc_for_hpo,
    best_n_estimators,
)


def _result(
    method: str,
    strategy: str,
    core: dict[str, Any],
    n_estimators: int,
    best_val_auc: float,
    runtime_s: float,
    extras: dict | None = None,
) -> dict[str, Any]:
    out = {
        "method": method,
        "strategy": strategy,
        "core_params": core,
        "n_estimators": int(n_estimators),
        "best_val_auc": float(best_val_auc),
        "runtime_seconds": float(runtime_s),
    }
    if extras:
        out.update(extras)
    return out


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def run_baseline(data: DatasetSplits) -> list[dict[str, Any]]:
    core = default_core_params()
    results = []

    # A: default params, fixed default n_estimators=100 (XGBoost default)
    t0 = time.perf_counter()
    model_a = train_strategy_a(data.X_train, data.y_train, core, n_estimators=100)
    va = val_auc_for_hpo(model_a, data.X_val, data.y_val)
    results.append(
        _result("baseline", "A_optimize_n_estimators", core, 100, va, time.perf_counter() - t0,
                extras={"note": "fixed default n_estimators=100, no search"})
    )

    # B: default params, high cap + early stopping
    t0 = time.perf_counter()
    model_b = train_strategy_b(data.X_train, data.y_train, data.X_val, data.y_val, core)
    n_b = best_n_estimators(model_b)
    vb = val_auc_for_hpo(model_b, data.X_val, data.y_val)
    results.append(
        _result("baseline", "B_early_stopping", core, n_b, vb, time.perf_counter() - t0)
    )
    return results


# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------
def run_optuna(data: DatasetSplits, strategy: str, n_trials: int = cfg.OPTUNA_TRIALS) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t0 = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        core = {
            "learning_rate": trial.suggest_float("learning_rate", cfg.HPARAM_LB[0], cfg.HPARAM_UB[0], log=True),
            "max_depth": trial.suggest_int("max_depth", int(cfg.HPARAM_LB[1]), int(cfg.HPARAM_UB[1])),
            "subsample": trial.suggest_float("subsample", cfg.HPARAM_LB[2], cfg.HPARAM_UB[2]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", cfg.HPARAM_LB[3], cfg.HPARAM_UB[3]),
            "min_child_weight": trial.suggest_float("min_child_weight", cfg.HPARAM_LB[4], cfg.HPARAM_UB[4]),
            "reg_lambda": trial.suggest_float("reg_lambda", cfg.HPARAM_LB[5], cfg.HPARAM_UB[5], log=True),
        }
        if strategy == "A_optimize_n_estimators":
            n_est = trial.suggest_int("n_estimators", cfg.N_EST_A_LOW, cfg.N_EST_A_HIGH)
            model = train_strategy_a(data.X_train, data.y_train, core, n_est)
            trial.set_user_attr("n_estimators", n_est)
        else:
            model = train_strategy_b(data.X_train, data.y_train, data.X_val, data.y_val, core)
            trial.set_user_attr("n_estimators", best_n_estimators(model))
        return val_auc_for_hpo(model, data.X_val, data.y_val)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    bp = study.best_params
    core = {k: bp[k] for k in cfg.HPARAM_NAMES}
    core["max_depth"] = int(core["max_depth"])
    n_est = int(study.best_trial.user_attrs["n_estimators"])
    if strategy == "A_optimize_n_estimators":
        n_est = int(bp["n_estimators"])

    return _result(
        "optuna",
        strategy,
        core,
        n_est,
        float(study.best_value),
        time.perf_counter() - t0,
        extras={"n_trials": n_trials},
    )


# ---------------------------------------------------------------------------
# mealpy PSO
# ---------------------------------------------------------------------------
def run_mealpy(data: DatasetSplits, strategy: str) -> dict[str, Any]:
    from mealpy import FloatVar, PSO

    t0 = time.perf_counter()
    is_a = strategy == "A_optimize_n_estimators"
    if is_a:
        lb = cfg.HPARAM_LB + [float(cfg.N_EST_A_LOW)]
        ub = cfg.HPARAM_UB + [float(cfg.N_EST_A_HIGH)]
    else:
        lb = list(cfg.HPARAM_LB)
        ub = list(cfg.HPARAM_UB)

    best_pack = {"val_auc": -1.0, "core": None, "n_estimators": None}

    def objective(solution: np.ndarray) -> float:
        core = decode_core_params(solution[:6])
        if is_a:
            n_est = int(round(float(solution[6])))
            n_est = int(np.clip(n_est, cfg.N_EST_A_LOW, cfg.N_EST_A_HIGH))
            model = train_strategy_a(data.X_train, data.y_train, core, n_est)
        else:
            model = train_strategy_b(data.X_train, data.y_train, data.X_val, data.y_val, core)
            n_est = best_n_estimators(model)
        auc = val_auc_for_hpo(model, data.X_val, data.y_val)
        if auc > best_pack["val_auc"]:
            best_pack["val_auc"] = auc
            best_pack["core"] = core
            best_pack["n_estimators"] = n_est
        return 1.0 - auc  # mealpy minimizes

    problem = {
        "bounds": FloatVar(lb=lb, ub=ub, name="xgb"),
        "obj_func": objective,
        "minmax": "min",
        "log_to": None,
        "name": f"mealpy_{strategy}",
    }
    model_pso = PSO.OriginalPSO(
        epoch=cfg.MEALPY_EPOCH,
        pop_size=cfg.MEALPY_POP,
        c1=2.05,
        c2=2.05,
        w=0.4,
    )
    g_best = model_pso.solve(problem)
    # Prefer tracked best (handles rounded n_estimators consistently)
    core = best_pack["core"] or decode_core_params(g_best.solution[:6])
    n_est = best_pack["n_estimators"]
    if n_est is None:
        if is_a:
            n_est = int(round(float(g_best.solution[6])))
        else:
            n_est = cfg.N_EST_B_CAP
    val_auc = best_pack["val_auc"] if best_pack["val_auc"] > 0 else 1.0 - float(g_best.target.fitness)

    return _result(
        "mealpy",
        strategy,
        core,
        int(n_est),
        float(val_auc),
        time.perf_counter() - t0,
        extras={"epoch": cfg.MEALPY_EPOCH, "pop_size": cfg.MEALPY_POP},
    )


# ---------------------------------------------------------------------------
# pyswarms PSO
# ---------------------------------------------------------------------------
def run_pyswarms(data: DatasetSplits, strategy: str) -> dict[str, Any]:
    import pyswarms as ps

    t0 = time.perf_counter()
    is_a = strategy == "A_optimize_n_estimators"
    if is_a:
        lb = np.array(cfg.HPARAM_LB + [float(cfg.N_EST_A_LOW)], dtype=float)
        ub = np.array(cfg.HPARAM_UB + [float(cfg.N_EST_A_HIGH)], dtype=float)
    else:
        lb = np.array(cfg.HPARAM_LB, dtype=float)
        ub = np.array(cfg.HPARAM_UB, dtype=float)

    best_pack = {"val_auc": -1.0, "core": None, "n_estimators": None}

    def objective(particles: np.ndarray) -> np.ndarray:
        costs = []
        for sol in particles:
            core = decode_core_params(sol[:6])
            if is_a:
                n_est = int(round(float(sol[6])))
                n_est = int(np.clip(n_est, cfg.N_EST_A_LOW, cfg.N_EST_A_HIGH))
                model = train_strategy_a(data.X_train, data.y_train, core, n_est)
            else:
                model = train_strategy_b(data.X_train, data.y_train, data.X_val, data.y_val, core)
                n_est = best_n_estimators(model)
            auc = val_auc_for_hpo(model, data.X_val, data.y_val)
            if auc > best_pack["val_auc"]:
                best_pack["val_auc"] = auc
                best_pack["core"] = core
                best_pack["n_estimators"] = n_est
            costs.append(1.0 - auc)
        return np.asarray(costs, dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=cfg.PYSWARMS_PARTICLES,
        dimensions=len(lb),
        options={"c1": 0.5, "c2": 0.3, "w": 0.9},
        bounds=(lb, ub),
    )
    best_cost, best_pos = optimizer.optimize(objective, iters=cfg.PYSWARMS_ITERS, verbose=False)

    core = best_pack["core"] or decode_core_params(best_pos[:6])
    n_est = best_pack["n_estimators"]
    if n_est is None:
        n_est = int(round(float(best_pos[6]))) if is_a else cfg.N_EST_B_CAP
    val_auc = best_pack["val_auc"] if best_pack["val_auc"] > 0 else 1.0 - float(best_cost)

    return _result(
        "pyswarms",
        strategy,
        core,
        int(n_est),
        float(val_auc),
        time.perf_counter() - t0,
        extras={"iters": cfg.PYSWARMS_ITERS, "n_particles": cfg.PYSWARMS_PARTICLES},
    )


STRATEGIES = ("A_optimize_n_estimators", "B_early_stopping")


def run_all_hpo(data: DatasetSplits, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Run baseline + all HPO methods for both strategies."""
    rows: list[dict[str, Any]] = []

    log("=== Baseline ===")
    for r in run_baseline(data):
        log(f"  {r['method']} / {r['strategy']}: val_auc={r['best_val_auc']:.5f} n_est={r['n_estimators']}")
        rows.append(r)

    for strategy in STRATEGIES:
        log(f"=== Optuna / {strategy} ===")
        r = run_optuna(data, strategy)
        log(f"  val_auc={r['best_val_auc']:.5f} n_est={r['n_estimators']} time={r['runtime_seconds']:.1f}s")
        rows.append(r)

        log(f"=== mealpy / {strategy} ===")
        r = run_mealpy(data, strategy)
        log(f"  val_auc={r['best_val_auc']:.5f} n_est={r['n_estimators']} time={r['runtime_seconds']:.1f}s")
        rows.append(r)

        log(f"=== pyswarms / {strategy} ===")
        r = run_pyswarms(data, strategy)
        log(f"  val_auc={r['best_val_auc']:.5f} n_est={r['n_estimators']} time={r['runtime_seconds']:.1f}s")
        rows.append(r)

    return rows
