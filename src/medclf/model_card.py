"""Generate a Markdown model card from an experiment result."""
from __future__ import annotations


def generate(result) -> str:
    name = result.best_model_name
    cfg = result.best_config
    m = result.best_metrics
    ds = result.data_summary
    base_recall = result.results["baseline"][name]["recall_pos"]
    smote_recall = result.results["smote"][name]["recall_pos"]

    def pct(x):
        return f"{x * 100:.1f}%"

    lines = [
        "# Model Card — Breast Cancer Malignancy Classifier",
        "",
        "## Overview",
        f"- **Best model:** `{name}` ({cfg} configuration)",
        "- **Task:** binary classification — predict whether a tumour is malignant (positive) or benign.",
        "- **Dataset:** Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn.",
        f"- **Features:** {ds['n_features']} real-valued cytology measurements.",
        f"- **Split:** {ds['n_train']} train / {ds['n_test']} test (stratified).",
        f"- **Simulated prevalence of malignant class in training:** {pct(ds['train_prevalence_malignant'])} "
        "(deliberately down-sampled to emulate screening-data imbalance).",
        "",
        "## Performance (test set)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {pct(m['accuracy'])} |",
        f"| F1 (macro) | {pct(m['f1_macro'])} |",
        f"| Precision (malignant) | {pct(m['precision_pos'])} |",
        f"| Recall (malignant) | {pct(m['recall_pos'])} |",
        f"| F1 (malignant) | {pct(m['f1_pos'])} |",
    ]
    if "roc_auc" in m:
        lines.append(f"| ROC-AUC | {m['roc_auc']:.3f} |")
    if "pr_auc" in m:
        lines.append(f"| PR-AUC | {m['pr_auc']:.3f} |")
    lines += [
        "",
        "## Effect of SMOTE on minority-class recall",
        "",
        f"For the best model, oversampling the minority (malignant) class with SMOTE changed "
        f"recall from **{pct(base_recall)}** (baseline) to **{pct(smote_recall)}** (SMOTE). "
        "Recall on the malignant class is the clinically critical metric because a false "
        "negative is a missed cancer.",
        "",
        "## Intended use & limitations",
        "- **Intended use:** educational demonstration of an end-to-end imbalanced-classification "
        "workflow (model comparison, resampling, calibrated thresholds, evaluation).",
        "- **NOT for clinical use.** This model is trained on a small public dataset and has not "
        "been validated on real patient populations. It must not be used for diagnosis.",
        "- **Bias/variance:** tree-based models can overfit the limited positive examples; metrics "
        "are reported on a held-out test set, but the small sample size means confidence intervals "
        "are wide.",
        "",
        "_This card was generated automatically by `medclf.model_card`._",
    ]
    return "\n".join(lines)
