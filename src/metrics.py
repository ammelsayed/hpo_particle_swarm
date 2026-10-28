"""Evaluation metrics: ROC-AUC and binned Asimov significance."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores))


def binned_asimov_significance(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 20,
    b_floor: float = 1e-6,
) -> float:
    """Combined binned Asimov significance (HEP-style).

    Treat class 1 as signal (s) and class 0 as background (b).
    Bin the BDT scores (predicted probability of class 1) into ``n_bins``
    equal-width bins over [0, 1].

    Per bin i with signal count s_i and background count b_i (after flooring
    background at ``b_floor`` to avoid log singularities):

        z_i^2 = 2 * ((s_i + b_i) * ln(1 + s_i / b_i) - s_i)

    Combined:

        Z = sqrt(sum_i z_i^2)

    Empty signal-and-background bins contribute zero.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    # Clip scores into [0, 1] for probability-style BDT scores
    scores = np.clip(scores, 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Digitize into bins 0..n_bins-1
    bin_idx = np.digitize(scores, edges[1:-1], right=False)

    z2_sum = 0.0
    for i in range(n_bins):
        mask = bin_idx == i
        if not np.any(mask):
            continue
        s = float(np.sum(y_true[mask] == 1))
        b = float(np.sum(y_true[mask] == 0))
        b = max(b, b_floor)
        if s <= 0.0:
            continue
        z2_sum += 2.0 * ((s + b) * np.log1p(s / b) - s)

    return float(np.sqrt(max(z2_sum, 0.0)))
