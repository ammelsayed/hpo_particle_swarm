"""Shared configuration for the HPO comparison study."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

RANDOM_SEED = 42

# Synthetic dataset size (large enough that GPU helps)
N_SAMPLES = 30_000
N_FEATURES = 50
N_INFORMATIVE = 30
N_REDUNDANT = 10

# Stratified split fractions: train / val / test
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20

# Strategy A: search n_estimators, no early stopping
N_EST_A_LOW = 50
N_EST_A_HIGH = 500

# Strategy B: high cap + early stopping on validation
N_EST_B_CAP = 2000
EARLY_STOPPING_ROUNDS = 50

# Shared continuous hyperparameter bounds (excluding n_estimators)
# order: learning_rate, max_depth, subsample, colsample_bytree,
#        min_child_weight, reg_lambda
HPARAM_LB = [0.01, 2.0, 0.50, 0.50, 1.0, 0.10]
HPARAM_UB = [0.30, 10.0, 1.00, 1.00, 10.0, 10.0]
HPARAM_NAMES = [
    "learning_rate",
    "max_depth",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_lambda",
]

# Budget (keep comparable across HPO methods)
OPTUNA_TRIALS = 40
MEALPY_EPOCH = 10
MEALPY_POP = 8
PYSWARMS_ITERS = 10
PYSWARMS_PARTICLES = 8

# Asimov binning on BDT scores (predicted probability of class 1)
ASIMOV_N_BINS = 20
ASIMOV_B_FLOOR = 1e-6

# XGBoost device (cuda if available; falls back handled in xgb_utils)
XGB_DEVICE = "cuda"
XGB_TREE_METHOD = "hist"
