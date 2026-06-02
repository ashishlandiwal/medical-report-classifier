"""Interactive diagnosis-exploration dashboard for non-technical stakeholders.

Run with:  streamlit run app/streamlit_app.py
(Requires a trained model — run `python -m medclf.train --no-mlflow` first.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medclf.data import load_data  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")
st.title("🩺 Breast Cancer Malignancy Classifier")
st.caption("Educational demo — not for clinical use.")

model_path = REPORTS / "best_model.joblib"
metrics_path = REPORTS / "metrics.json"

if not model_path.exists():
    st.warning("No trained model found. Run `python -m medclf.train --no-mlflow` first.")
    st.stop()

model = joblib.load(model_path)
metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
ds = load_data()

left, right = st.columns([2, 1])

with right:
    st.subheader("Best model")
    st.write(f"**{metrics.get('best_model_name', '?')}** ({metrics.get('best_config', '?')})")
    bm = metrics.get("best_metrics", {})
    if bm:
        st.metric("Recall (malignant)", f"{bm.get('recall_pos', 0) * 100:.1f}%")
        st.metric("F1 (macro)", f"{bm.get('f1_macro', 0) * 100:.1f}%")
        if "roc_auc" in bm:
            st.metric("ROC-AUC", f"{bm['roc_auc']:.3f}")
    for img in ("confusion_matrix.png", "pr_curve.png"):
        p = REPORTS / img
        if p.exists():
            st.image(str(p))

with left:
    st.subheader("Try a patient sample")
    n = len(ds.X_test)
    idx = st.slider("Test-set sample index", 0, n - 1, 0)
    sample = ds.X_test.iloc[[idx]]
    true_label = ds.class_names[int(ds.y_test[idx])]

    pred = int(model.predict(sample)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(sample)[0, 1])

    pred_label = ds.class_names[pred]
    st.write(f"**Predicted:** {pred_label}" + (f"  ·  P(malignant) = {proba:.2%}" if proba is not None else ""))
    st.write(f"**Actual:** {true_label}")
    if pred_label == true_label:
        st.success("Correct prediction ✅")
    else:
        st.error("Incorrect prediction ❌")

    with st.expander("Feature values for this sample"):
        st.dataframe(sample.T.rename(columns={sample.index[0]: "value"}))
