# Model Card — Breast Cancer Malignancy Classifier

## Overview
- **Best model:** `knn` (smote configuration)
- **Task:** binary classification — predict whether a tumour is malignant (positive) or benign.
- **Dataset:** Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn.
- **Features:** 30 real-valued cytology measurements.
- **Split:** 324 train / 82 test (stratified).
- **Simulated prevalence of malignant class in training:** 12.0% (deliberately down-sampled to emulate screening-data imbalance).

## Performance (test set)

| Metric | Value |
|---|---|
| Accuracy | 97.6% |
| F1 (macro) | 93.8% |
| Precision (malignant) | 100.0% |
| Recall (malignant) | 80.0% |
| F1 (malignant) | 88.9% |
| ROC-AUC | 0.900 |
| PR-AUC | 0.824 |

## Effect of SMOTE on minority-class recall

For the best model, oversampling the minority (malignant) class with SMOTE changed recall from **70.0%** (baseline) to **80.0%** (SMOTE). Recall on the malignant class is the clinically critical metric because a false negative is a missed cancer.

## Intended use & limitations
- **Intended use:** educational demonstration of an end-to-end imbalanced-classification workflow (model comparison, resampling, calibrated thresholds, evaluation).
- **NOT for clinical use.** This model is trained on a small public dataset and has not been validated on real patient populations. It must not be used for diagnosis.
- **Bias/variance:** tree-based models can overfit the limited positive examples; metrics are reported on a held-out test set, but the small sample size means confidence intervals are wide.

_This card was generated automatically by `medclf.model_card`._