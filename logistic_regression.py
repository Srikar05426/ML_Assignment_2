"""Logistic regression with training-only feature standardization."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(random_state=42):
	return make_pipeline(
		StandardScaler(),
		LogisticRegression(max_iter=2000, random_state=random_state),
	)
