"""Synthetic data generation with a strict train / validation / test split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from . import config as cfg


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def make_splits(seed: int = cfg.RANDOM_SEED) -> DatasetSplits:
    """Create stratified train/val/test. Test is never used for fitting or HPO."""
    X, y = make_classification(
        n_samples=cfg.N_SAMPLES,
        n_features=cfg.N_FEATURES,
        n_informative=cfg.N_INFORMATIVE,
        n_redundant=cfg.N_REDUNDANT,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.02,
        random_state=seed,
    )
    # First peel off test (20%), then split remaining into train (60%) and val (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=cfg.TEST_FRAC,
        random_state=seed,
        stratify=y,
    )
    val_frac_of_temp = cfg.VAL_FRAC / (cfg.TRAIN_FRAC + cfg.VAL_FRAC)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_frac_of_temp,
        random_state=seed,
        stratify=y_temp,
    )
    return DatasetSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
