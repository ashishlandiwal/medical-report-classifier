"""The six classifiers compared in this project."""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    """Return the model zoo keyed by short name (deterministic where applicable)."""
    return {
        "logistic_regression": LogisticRegression(max_iter=5000),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "svm": SVC(probability=True, random_state=random_state),
    }
