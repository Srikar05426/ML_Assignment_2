"""Streamlit evaluation dashboard for saved mobile-price classifiers."""

from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from classification import calculate_metrics


ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
CLASS_NAMES = ["Low (0)", "Medium (1)", "High (2)", "Very High (3)"]
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def validate_test_data(data, feature_names):
    if data.empty:
        raise ValueError("The test CSV contains no rows.")
    required = set(feature_names) | {"price_range"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    unexpected = set(data.columns) - required - {"id", "source_row"}
    if unexpected:
        raise ValueError("Unexpected columns: " + ", ".join(sorted(unexpected)))
    values = data.loc[:, list(feature_names) + ["price_range"]]
    if values.isna().any().any():
        raise ValueError("The test CSV contains missing values.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in values.dtypes):
        raise ValueError("All features and price_range must be numeric.")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("The test CSV contains infinite values.")
    if not data["price_range"].isin([0, 1, 2, 3]).all():
        raise ValueError("price_range must contain only class labels 0, 1, 2, or 3.")
    return data.loc[:, feature_names], data["price_range"].astype(int)


@st.cache_resource
def load_models(signatures):
    return {
        name: joblib.load(ROOT / "model" / filename)
        for name, filename, modified, size in signatures
    }


@st.cache_data
def evaluate_test_data(data, signatures):
    models = load_models(signatures)
    features, target = validate_test_data(
        data, list(next(iter(models.values())).feature_names_in_)
    )
    scores = []
    details = {}
    for name, model in models.items():
        predicted = model.predict(features)
        probabilities = model.predict_proba(features)
        labels = model.classes_
        scores.append({"Model": name, **calculate_metrics(target, predicted, probabilities, labels)})
        predictions = data.copy()
        predictions["predicted_price_range"] = predicted
        predictions["confidence"] = probabilities.max(axis=1)
        for index, label in enumerate(labels):
            predictions[f"probability_{label}"] = probabilities[:, index]
        details[name] = {
            "confusion": confusion_matrix(target, predicted, labels=labels),
            "report": pd.DataFrame(classification_report(
                target, predicted, labels=labels, target_names=CLASS_NAMES,
                output_dict=True, zero_division=0,
            )).transpose(),
            "predictions": predictions,
        }
    return pd.DataFrame(scores), details


def main():
    st.set_page_config(page_title="Mobile Price Lab", page_icon=":material/smartphone:", layout="wide")
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
        html, body, [class*="css"], .stApp, h1, h2, h3 {
            font-family: 'IBM Plex Sans', 'Trebuchet MS', sans-serif;
            letter-spacing: 0 !important;
        }
        .stApp { background: linear-gradient(120deg, #f8faf9 0%, #ffffff 70%, #f5f7fa 100%); }
        h1 { font-size: 2.4rem !important; color: #172a26; }
        h2, h3 { color: #172a26; }
        [data-testid="stSidebar"] { background: #edf2ef; border-right: 1px solid #dce5df; }
        [data-testid="stMetric"] { border-top: 3px solid #19705c; padding-top: 12px; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; }
        .block-container { max-width: 1440px; padding-top: 2rem; }
        @media (max-width: 640px) {
            h1 { font-size: 1.9rem !important; }
            .block-container { padding: 1rem; }
        }
        </style>
    """, unsafe_allow_html=True)

    try:
        signatures = tuple(
            (name, filename, (ROOT / "model" / filename).stat().st_mtime_ns,
             (ROOT / "model" / filename).stat().st_size)
            for name, filename in MODEL_FILES.items()
        )
        models = load_models(signatures)
        bundled = (ROOT / "test_data.csv").read_bytes()
    except (OSError, ValueError, ImportError) as error:
        st.error(f"Saved models or bundled test data are unavailable: {error}")
        st.stop()

    with st.sidebar:
        st.header("Evaluation", divider="green")
        selected = st.selectbox("Classifier", list(MODEL_FILES))
        uploaded = st.file_uploader("Test dataset", type=["csv"])
        st.download_button(
            "Download test CSV", bundled, "test_data.csv", "text/csv", icon=":material/download:",
        )
        st.divider()
        st.caption("Mobile Price Classification / Kaggle")
        st.caption("Training: 1,600 rows | Holdout: 400 rows")

    try:
        data = pd.read_csv(uploaded if uploaded is not None else BytesIO(bundled))
        features, target = validate_test_data(
            data, list(next(iter(models.values())).feature_names_in_)
        )
        with st.spinner("Evaluating saved classifiers..."):
            scores, details = evaluate_test_data(data, signatures)
    except (ValueError, pd.errors.ParserError, UnicodeError) as error:
        st.error(f"Invalid test dataset: {error}")
        st.stop()

    st.title("Mobile Price Lab")
    st.caption(f"{uploaded.name if uploaded is not None else 'Bundled holdout'} | "
               f"{len(data):,} test rows | {features.shape[1]} features | {target.nunique()} classes")
    evaluation, comparison, dataset = st.tabs(["Model Evaluation", "Model Comparison", "Test Data"])

    with evaluation:
        st.subheader(selected)
        row = scores.set_index("Model").loc[selected]
        for columns, metric_group in [(st.columns(3), METRICS[:3]), (st.columns(3), METRICS[3:])]:
            for column, metric in zip(columns, metric_group):
                value = row[metric]
                column.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")
        st.caption("Precision, recall, F1: macro average over four classes. AUC: macro one-vs-rest ROC-AUC.")
        if pd.isna(row["AUC"]):
            st.warning("AUC is unavailable because the test data does not contain all four classes.")
        st.subheader("Confusion Matrix")
        figure = px.imshow(
            details[selected]["confusion"], x=CLASS_NAMES, y=CLASS_NAMES,
            text_auto=True, color_continuous_scale=["#f0f5f2", "#19705c"],
            labels={"x": "Predicted class", "y": "Actual class", "color": "Count"},
            aspect="auto",
        )
        figure.update_layout(height=390, margin=dict(l=0, r=0, t=10, b=0), font_family="IBM Plex Sans")
        st.plotly_chart(figure, width="stretch")
        st.subheader("Classification Report")
        st.dataframe(details[selected]["report"], width="stretch")
        st.download_button(
            "Download predictions", details[selected]["predictions"].to_csv(index=False),
            f"{MODEL_FILES[selected].removesuffix('.joblib')}_predictions.csv", "text/csv",
            icon=":material/download:",
        )

    with comparison:
        st.subheader("All Five Classifiers")
        ordered = scores.sort_values("F1", ascending=False).reset_index(drop=True)
        st.dataframe(ordered.style.format({metric: "{:.4f}" for metric in METRICS}),
                     hide_index=True, width="stretch")
        chart_metric = st.selectbox("Comparison metric", METRICS)
        chart = px.bar(
            scores.sort_values(chart_metric), x=chart_metric, y="Model", orientation="h",
            color="Model", color_discrete_sequence=["#19705c", "#da745b", "#4a75a3", "#b59a31", "#64766b"],
        )
        chart.update_layout(showlegend=False, height=380, margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_range=[-1 if chart_metric == "MCC" else 0, 1])
        st.plotly_chart(chart, width="stretch")
        st.download_button("Download metrics", scores.to_csv(index=False), "metrics.csv", "text/csv",
                           icon=":material/download:")

    with dataset:
        st.subheader("Test Dataset")
        counts = target.value_counts().reindex([0, 1, 2, 3], fill_value=0)
        st.dataframe(pd.DataFrame({"Price range": CLASS_NAMES, "Instances": counts.to_numpy()}),
                     hide_index=True, width="stretch")
        st.dataframe(data, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()