"""Train all six models with and without SMOTE, pick the best, and save artifacts."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import evaluate
from .data import Dataset, load_data
from .models import get_models

CONFIGS = {"baseline": False, "smote": True}


@dataclass
class ExperimentResult:
    results: dict
    best_model_name: str
    best_config: str
    best_metrics: dict
    data_summary: dict
    best_estimator: object = field(default=None, repr=False)
    dataset: Dataset | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        return {
            "results": self.results,
            "best_model_name": self.best_model_name,
            "best_config": self.best_config,
            "best_metrics": self.best_metrics,
            "data_summary": self.data_summary,
        }


def _build_pipeline(model, use_smote: bool, random_state: int):
    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state, k_neighbors=3)))
        return ImbPipeline(steps + [("clf", model)])
    return Pipeline(steps + [("clf", model)])


def _positive_proba(fitted, X) -> np.ndarray | None:
    if hasattr(fitted, "predict_proba"):
        return fitted.predict_proba(X)[:, 1]
    if hasattr(fitted, "decision_function"):
        scores = fitted.decision_function(X)
        span = scores.max() - scores.min()
        return (scores - scores.min()) / (span + 1e-9)
    return None


def run_experiment(
    dataset: Dataset | None = None,
    random_state: int = 42,
    use_mlflow: bool = False,
) -> ExperimentResult:
    ds = dataset or load_data(random_state=random_state)

    mlflow = None
    if use_mlflow:
        try:
            import mlflow as _mlflow

            mlflow = _mlflow
            mlflow.set_experiment("medical-report-classifier")
        except Exception:
            mlflow = None

    results: dict[str, dict] = {}
    best_score, best_name, best_cfg, best_metrics, best_est = -1.0, "", "", {}, None

    for cfg, use_smote in CONFIGS.items():
        results[cfg] = {}
        for name, model in get_models(random_state).items():
            pipe = _build_pipeline(model, use_smote, random_state)
            pipe.fit(ds.X_train, ds.y_train)
            y_pred = pipe.predict(ds.X_test)
            y_prob = _positive_proba(pipe, ds.X_test)
            metrics = evaluate.compute_metrics(ds.y_test, y_pred, y_prob)
            results[cfg][name] = metrics

            if mlflow is not None:
                with mlflow.start_run(run_name=f"{name}-{cfg}"):
                    mlflow.log_params({"model": name, "smote": use_smote})
                    mlflow.log_metrics(metrics)

            if metrics["f1_macro"] > best_score:
                best_score, best_name, best_cfg = metrics["f1_macro"], name, cfg
                best_metrics, best_est = metrics, pipe

    data_summary = {
        "n_train": int(len(ds.y_train)),
        "n_test": int(len(ds.y_test)),
        "n_features": int(ds.n_features),
        "train_prevalence_malignant": round(ds.train_prevalence, 4),
        "class_names": ds.class_names,
    }
    return ExperimentResult(
        results=results,
        best_model_name=best_name,
        best_config=best_cfg,
        best_metrics=best_metrics,
        data_summary=data_summary,
        best_estimator=best_est,
        dataset=ds,
    )


def _smote_effect(result: ExperimentResult) -> dict:
    """Positive-class recall, baseline vs SMOTE, for the best model — the headline
    demonstration that SMOTE recovers minority-class sensitivity."""
    name = result.best_model_name
    base = result.results["baseline"][name]["recall_pos"]
    smote = result.results["smote"][name]["recall_pos"]
    return {"model": name, "recall_baseline": base, "recall_smote": smote}


def save_artifacts(result: ExperimentResult, output_dir: str | Path) -> None:
    from .model_card import generate as generate_card

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "metrics.json").write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    if result.best_estimator is not None:
        joblib.dump(result.best_estimator, out / "best_model.joblib")

    ds = result.dataset
    if ds is not None and result.best_estimator is not None:
        y_pred = result.best_estimator.predict(ds.X_test)
        y_prob = _positive_proba(result.best_estimator, ds.X_test)
        evaluate.plot_confusion_matrix(ds.y_test, y_pred, ds.class_names, out / "confusion_matrix.png")
        if y_prob is not None:
            evaluate.plot_pr_curve(ds.y_test, y_prob, out / "pr_curve.png")

    (out / "model_card.md").write_text(generate_card(result), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports", help="directory for artifacts")
    parser.add_argument("--imbalance", type=float, default=0.12, help="malignant prevalence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    args = parser.parse_args()

    ds = load_data(imbalance_ratio=args.imbalance, random_state=args.seed)
    result = run_experiment(ds, random_state=args.seed, use_mlflow=not args.no_mlflow)
    save_artifacts(result, args.output)

    effect = _smote_effect(result)
    print("\n=== Per-model F1 (macro) ===")
    for cfg in CONFIGS:
        row = ", ".join(f"{m}={result.results[cfg][m]['f1_macro']:.3f}" for m in result.results[cfg])
        print(f"  {cfg:8s}: {row}")
    print(f"\nBest: {result.best_model_name} ({result.best_config})  ->  {result.best_metrics}")
    print(
        f"SMOTE effect on '{effect['model']}' malignant recall: "
        f"{effect['recall_baseline']:.3f} -> {effect['recall_smote']:.3f}"
    )
    print(f"Artifacts written to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
