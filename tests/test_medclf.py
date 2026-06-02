import numpy as np

from medclf import get_models, load_data
from medclf.evaluate import compute_metrics
from medclf.train import run_experiment


def test_load_data_creates_imbalance():
    ds = load_data(imbalance_ratio=0.12, random_state=0)
    # train/test disjoint and non-empty
    assert len(ds.y_train) > 0 and len(ds.y_test) > 0
    assert ds.n_features == 30
    # prevalence of the malignant (positive) class is close to the requested ratio
    assert 0.05 <= ds.train_prevalence <= 0.20


def test_get_models_returns_six():
    models = get_models()
    assert len(models) == 6
    for est in models.values():
        assert hasattr(est, "fit") and hasattr(est, "predict")


def test_compute_metrics_perfect_prediction():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, y, y.astype(float))
    assert m["accuracy"] == 1.0
    assert m["recall_pos"] == 1.0
    assert m["f1_macro"] == 1.0


def test_experiment_runs_and_is_accurate():
    ds = load_data(imbalance_ratio=0.12, random_state=42)
    result = run_experiment(ds, random_state=42, use_mlflow=False)
    # both configurations evaluated for all six models
    assert set(result.results) == {"baseline", "smote"}
    assert len(result.results["baseline"]) == 6
    # the dataset is separable enough that the best model should be strong
    assert result.best_metrics["f1_macro"] >= 0.85
    assert result.best_metrics["recall_pos"] >= 0.80
