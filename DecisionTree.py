"""Decision tree classifier for mobile price ranges."""

from sklearn.tree import DecisionTreeClassifier


def build_model(random_state=42):
	return DecisionTreeClassifier(random_state=random_state)
