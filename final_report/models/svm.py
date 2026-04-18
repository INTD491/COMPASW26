"""
Support Vector Machine for COMPAS recidivism prediction.
Requires feature scaling (handled by train.py).
"""

from sklearn.svm import SVC


def build_model():
    return SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,  # needed for ROC-AUC
        random_state=42,
    )


NEEDS_SCALING = True
