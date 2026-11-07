"""Compare five classifiers on Kaggle's Mobile Price Classification dataset."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from DecisionTree import build_model as build_decision_tree
from KNN import build_model as build_knn
from NaiveBayes import build_model as build_naive_bayes
from SVM import build_model as build_svm
from logistic_regression import build_model as build_logistic_regression


ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42


def calculate_metrics(target, predicted, probabilities, labels):
    auc = float("nan")
    if set(target.unique()) == set(labels):
        auc = roc_auc_score(
            target, probabilities, labels=labels, multi_class="ovr", average="macro"
        )
    return {
        "Accuracy": accuracy_score(target, predicted),
        "AUC": auc,
        "Precision": precision_score(target, predicted, average="macro", labels=labels, zero_division=0),
        "Recall": recall_score(target, predicted, average="macro", labels=labels, zero_division=0),
        "F1": f1_score(target, predicted, average="macro", labels=labels, zero_division=0),
        "MCC": matthews_corrcoef(target, predicted),
    }


def evaluate_models(data_path, output_dir, model_dir=ROOT / "model"):
    data = pd.read_csv(data_path)
    if "price_range" not in data.columns:
        raise ValueError("Use labeled train.csv containing the price_range column.")

    features = data.drop(columns=["price_range", "id"], errors="ignore")
    target = data["price_range"]
    if len(data) < 500 or features.shape[1] < 12:
        raise ValueError("The dataset must contain at least 500 rows and 12 features.")
    if data.isna().any().any():
        raise ValueError("Missing values found; clean the dataset before training.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("This implementation expects numerical input features.")
    if set(target.unique()) != {0, 1, 2, 3}:
        raise ValueError("Expected the four Mobile Price classes: 0, 1, 2, 3.")

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    models = {
        "Logistic Regression": build_logistic_regression(RANDOM_STATE),
        "Decision Tree": build_decision_tree(RANDOM_STATE),
        "K-Nearest Neighbors": build_knn(),
        "Gaussian Naive Bayes": build_naive_bayes(),
        "Support Vector Machine": build_svm(RANDOM_STATE),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data.loc[train_features.index].to_csv(output_dir / "training_data.csv", index=False)
    data.loc[test_features.index].to_csv(ROOT / "test_data.csv", index=False)
    scores = []
    reports = {}
    predictions = pd.DataFrame({"actual": test_target}, index=test_target.index)

    for name, estimator in models.items():
        model = clone(estimator)
        model.fit(train_features, train_target)
        predicted = model.predict(test_features)
        probabilities = model.predict_proba(test_features)
        labels = model.classes_
        scores.append({"Model": name, **calculate_metrics(test_target, predicted, probabilities, labels)})
        reports[name] = classification_report(
            test_target, predicted, output_dict=True, zero_division=0
        )
        predictions[name] = predicted
        filename = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(model, model_dir / f"{filename}.joblib")
        pd.DataFrame(
            confusion_matrix(test_target, predicted, labels=labels),
            index=pd.Index(labels, name="actual"),
            columns=pd.Index(labels, name="predicted"),
        ).to_csv(output_dir / f"{filename}_confusion_matrix.csv")

    results = pd.DataFrame(scores).sort_values("F1", ascending=False)
    results.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "holdout_predictions.csv", index_label="source_row")
    (output_dir / "classification_reports.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    print(f"Dataset: {len(data)} rows, {features.shape[1]} features")
    print(f"Training rows: {len(train_target)}; holdout rows: {len(test_target)}")
    print("\n" + results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nEvaluation files: {output_dir.resolve()}")
    print(f"Saved trained models: {model_dir.resolve()}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "model")
    args = parser.parse_args()
    evaluate_models(args.data, args.output, args.model_dir)


if __name__ == "__main__":
    main()