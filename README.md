# Mobile Price Classification

## a. Problem Statement

Classify mobile phones into four price ranges using their specifications.
Compare five classification models using six evaluation metrics and display
the results in a Streamlit app.

## b. Dataset Description

- Source: [Kaggle Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification).
- Dataset: 2,000 labeled instances and 20 numerical features, with no missing values.
- Features include RAM, battery power, storage, camera resolution, and screen dimensions.
- Target: `price_range`: 0 (low), 1 (medium), 2 (high), and 3 (very high), with 500 rows per class.
- Meets the minimum of 500 instances and 12 features before splitting.

### Training and Testing

All models use the same stratified 80/20 split with `random_state=42`:
1,600 training rows and 400 test rows. Scaling is fitted only on training data
and saved with the models. The SVM uses five-fold probability calibration
within the training set.

- [data/train.csv](data/train.csv): original labeled dataset.
- [results/training_data.csv](results/training_data.csv): training split.
- [test_data.csv](test_data.csv): labeled test split used for evaluation and app uploads.

Kaggle's separate [data/test.csv](data/test.csv) has no target labels, so it is
not used to calculate metrics.

## c. GitHub Repository Link

Repository URL: `https://github.com/Srikar05426/ML_Assignment_2`

## d. Models Used

| Model | Configuration |
|---|---|
| Logistic Regression | StandardScaler; `max_iter=2000`; seed 42 |
| Decision Tree | Default tree; seed 42 |
| K-Nearest Neighbors | StandardScaler; 5 neighbors; uniform distance voting |
| Gaussian Naive Bayes | StandardScaler; GaussianNB with default smoothing |
| Support Vector Machine | StandardScaler; RBF kernel; default C and gamma; five-fold probability calibration |

### Evaluation Comparison

Results on the same 400 held-out instances:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9650 | 0.9987 | 0.9650 | 0.9650 | 0.9650 | 0.9534 |
| Decision Tree | 0.8300 | 0.8867 | 0.8319 | 0.8300 | 0.8302 | 0.7738 |
| KNN | 0.5000 | 0.7697 | 0.5211 | 0.5000 | 0.5054 | 0.3350 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Support Vector Machine | 0.8950 | 0.9580 | 0.8969 | 0.8950 | 0.8956 | 0.8602 |

Precision, recall, and F1 use macro averaging. AUC is macro one-vs-rest ROC-AUC
using class probabilities. AUC is unavailable when an uploaded test set does
not contain all four classes. MCC uses the multiclass calculation.

### Observations

| ML Model Name | Observation About Model Performance |
|---|---|
| Logistic Regression | Highest scores across all six metrics, with 96.50% accuracy and 0.9650 F1. |
| Decision Tree | Achieved 83.00% accuracy, performing better than Naive Bayes and KNN on this split. |
| KNN | Lowest accuracy at 50.00%. The default five-neighbor model was less effective with these features. |
| Naive Bayes | Achieved 81.00% accuracy and 0.8105 F1, below Decision Tree, SVM, and Logistic Regression. |
| Support Vector Machine | Second-best accuracy at 89.50%, with 0.8956 F1. |
| Overall Winner | Logistic Regression performed best on this test split. |

## Local Setup

Saved models were trained with Python 3.14.3 and scikit-learn 1.9.1.
Use the pinned dependencies in [requirements.txt](requirements.txt).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Reproduce Training and Testing

```bash
python classification.py
```

This saves the five trained models, labeled test split, and evaluation results.

### Start the Streamlit App

```bash
python -m streamlit run app.py
```

Upload a test CSV containing all 20 input features and `price_range`, or use the
bundled test data. Select a model to view its six metrics, confusion matrix, and
classification report. The comparison tab shows results for all five models.
The app loads saved models without retraining on uploaded data.

## Project Files

```text
app.py                    Streamlit test-evaluation dashboard
classification.py         Shared training, testing, metrics, and model saving
logistic_regression.py    Logistic Regression pipeline
DecisionTree.py           Decision Tree classifier
KNN.py                    KNN pipeline
NaiveBayes.py             Gaussian Naive Bayes pipeline
SVM.py                    Calibrated SVM pipeline
requirements.txt          Pinned runtime dependencies
test_data.csv             Labeled held-out test split
data/                     Original Kaggle CSV files
model/                    Five fitted .joblib models (including preprocessing)
results/                  Metrics, reports, matrices, training split, predictions
.streamlit/config.toml    Theme and upload limit
```
