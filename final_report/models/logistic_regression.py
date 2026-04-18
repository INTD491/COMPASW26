"""
Logistic Regression for COMPAS recidivism prediction.
Requires feature scaling (handled by train.py).
"""

from sklearn.linear_model import LogisticRegression


def build_model():
    return LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
    )


NEEDS_SCALING = True
