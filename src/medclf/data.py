"""Dataset loading and a controlled class-imbalance simulation.

Uses the Breast Cancer Wisconsin (Diagnostic) dataset that ships with scikit-learn
(569 samples, 30 real-valued features). We relabel so that ``1 = malignant`` — the
clinically important "positive" case we must not miss — and optionally **down-sample
the malignant class** to emulate the severe class imbalance common in real screening
data, so we can demonstrate the effect of SMOTE honestly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

POSITIVE_LABEL = 1  # malignant
CLASS_NAMES = ["benign", "malignant"]


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))

    @property
    def train_prevalence(self) -> float:
        return float(np.mean(self.y_train))

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def load_data(
    test_size: float = 0.2,
    imbalance_ratio: float | None = 0.12,
    random_state: int = 42,
) -> Dataset:
    """Load the dataset and split it (stratified).

    Args:
        imbalance_ratio: target prevalence of the malignant (positive) class after
            down-sampling. Pass ``None`` to keep the natural class balance.
    """
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    # sklearn encodes 0 = malignant, 1 = benign. Flip so 1 = malignant (positive).
    y = (raw.target.to_numpy() == 0).astype(int)

    if imbalance_ratio is not None:
        rng = np.random.RandomState(random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_neg = len(neg_idx)
        pos_keep = int(round(imbalance_ratio / (1.0 - imbalance_ratio) * n_neg))
        pos_keep = max(2, min(pos_keep, len(pos_idx)))
        keep_pos = rng.choice(pos_idx, size=pos_keep, replace=False)
        keep = np.sort(np.concatenate([neg_idx, keep_pos]))
        X = X.iloc[keep].reset_index(drop=True)
        y = y[keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X.columns),
    )
