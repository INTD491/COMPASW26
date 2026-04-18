"""
Random Forest for COMPAS recidivism prediction.
Does not require feature scaling.
"""

from sklearn.ensemble import RandomForestClassifier


def build_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )


NEEDS_SCALING = False
