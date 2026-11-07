"""K-nearest neighbors with training-only feature standardization."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
	return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
