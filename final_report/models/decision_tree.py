"""
Decision Tree for COMPAS recidivism prediction.
Does not require feature scaling.
"""

from sklearn.tree import DecisionTreeClassifier


def build_model():
    return DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
    )


NEEDS_SCALING = False
