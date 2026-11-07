"""RBF support vector classifier with probability estimates."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(random_state=42):
	return CalibratedClassifierCV(
		make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=random_state)),
		cv=5,
		ensemble=False,
	)
