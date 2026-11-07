"""Gaussian naive Bayes for continuous numerical features."""

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
	return make_pipeline(StandardScaler(), GaussianNB())
