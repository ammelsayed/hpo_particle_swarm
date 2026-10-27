# Particle Swarm Optimization for XGBoost Hyperparameters

Hyperparameter search for **XGBoost** using the classic **Particle Swarm Optimization (PSO)** algorithm from an existing library — not a hand-rolled PSO.

## Why `mealpy`?

| Package | PSO? | Notes |
|---------|------|-------|
| **mealpy** (chosen) | Yes — `PSO.OriginalPSO` | Actively maintained, clear docs, `FloatVar` bounds API, classical Kennedy & Eberhart (1995) PSO. Easy to wrap any objective (including XGBoost). |
| pyswarms | Yes — `GlobalBestPSO` | Solid PSO toolkit; objective must return a vector of costs (one per particle). Also a good choice. |
| scikit-opt (`sko`) | Yes — `sko.PSO` | Simple API; less actively maintained than mealpy. |
| Optuna / Hyperopt | No native PSO | Bayesian / TPE style — out of scope for this study. |

We picked **mealpy** because it documents PSO explicitly, is maintained, and lets us plug in a custom fitness function that trains XGBoost and returns a scalar loss (`1 - AUC`).

Installed versions are recorded in `results.json` after a run.

## Environment

- Python venv: `/data/ammelsayed/venv`
- GPU: NVIDIA A10 × 4 (driver 565.57.01 → CUDA 12.7 capability)
- XGBoost must be the **CUDA 12** wheel on this machine (`xgboost-cu12`), because the default `xgboost` PyPI wheel is CUDA 13 and silently falls back to CPU here.
- Extra dependency: `mealpy`

```bash
source /data/ammelsayed/venv/bin/activate
pip install mealpy
pip uninstall -y xgboost xgboost-cu12
pip install xgboost-cu12
# mealpy may pin an old numpy; restore 2.x for the rest of this env:
pip install 'numpy>=2.1'
```

Verify GPU:

```bash
python -c "import xgboost as xgb; print(xgb.build_info()['CUDA_VERSION'])"
# expect something like [12, 9], and no 'No visible GPU' warnings when training
```

## Data

Synthetic binary classification via `sklearn.datasets.make_classification`:

- 20,000 samples, 60 features (40 informative)
- 80/20 stratified train/test split
- Metric: **ROC-AUC** on the hold-out set

## Hyperparameters searched

| Parameter | Lower | Upper |
|-----------|-------|-------|
| `learning_rate` | 0.01 | 0.30 |
| `max_depth` | 2 | 10 (rounded to int) |
| `subsample` | 0.50 | 1.00 |
| `colsample_bytree` | 0.50 | 1.00 |
| `min_child_weight` | 1.0 | 10.0 |
| `reg_lambda` | 0.1 | 10.0 |

Fixed: `n_estimators=200`, `objective=binary:logistic`, `tree_method=hist`, `device=cuda`.

PSO settings: 15 epochs × 12 particles (180 fitness evaluations).

## How to run

```bash
source /data/ammelsayed/venv/bin/activate
cd /data/ammelsayed/particle_swarm
python optimize_xgboost_pso.py
```

Output:

- Console progress (AUC per evaluation batch)
- `results.json` with best params, AUC, runtime, package versions

## Results

See `results.json` after running. Summary fields:

- `best_params` — tuned XGBoost hyperparameters
- `best_auc_retrain` — hold-out ROC-AUC after retrain with best params
- `runtime_seconds` — wall time for the PSO search
- `device` — should be `"cuda"`
- `package` / `package_version` — mealpy details

## Run completed (2026-10-28, Asia/Shanghai)

- **Package:** mealpy `3.0.3` — `PSO.OriginalPSO`
- **XGBoost:** `3.4.1` via `xgboost-cu12` (CUDA_VERSION `[12, 9]`), `device=cuda` on NVIDIA A10
- **Data:** `make_classification` 20k×60, hold-out AUC
- **Best AUC:** `0.988118`
- **Best params:**
  - learning_rate ≈ 0.1601
  - max_depth = 9
  - subsample ≈ 0.7218
  - colsample_bytree ≈ 0.9932
  - min_child_weight ≈ 4.51
  - reg_lambda ≈ 3.43
  - n_estimators = 200
- **Runtime:** ~780 s (~13 min) for 15 epochs × 12 particles
