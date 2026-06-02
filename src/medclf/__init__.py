"""Medical diagnosis classifier: 6 models, class-imbalance handling, model card.

Public API:
    load_data(...)      -> Dataset (train/test split with simulated imbalance)
    get_models(...)     -> dict[str, estimator]
    medclf.train.run_experiment(...) -> ExperimentResult (metrics, with/without SMOTE)
"""
from .data import Dataset, load_data
from .models import get_models

__all__ = ["load_data", "Dataset", "get_models"]
__version__ = "0.1.0"
