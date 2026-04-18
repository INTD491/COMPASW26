"""
Fairlearn EqualizedOdds wrapper around Random Forest.
Trains a randomized classifier that constrains FPR/FNR gaps across racial groups.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds


class FairlearnEO:
    def __init__(self):
        base = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            min_samples_leaf=4, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        self.model = ExponentiatedGradient(
            estimator=base,
            constraints=EqualizedOdds(difference_bound=0.05),
            eps=0.05, max_iter=50,
        )

    def fit(self, X, y, sensitive_features):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.model.fit(X_df, y, sensitive_features=sensitive_features)
        return self

    def predict(self, X, sensitive_features=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return np.asarray(self.model.predict(X_df), dtype=int)

    def predict_proba(self, X, sensitive_features=None):
        # ExponentiatedGradient is a randomized classifier; no calibrated proba.
        # Return degenerate proba from hard predictions so AUC is defined.
        preds = self.predict(X)
        proba = np.zeros((len(preds), 2))
        proba[np.arange(len(preds)), preds] = 1.0
        return proba


def build_model():
    return FairlearnEO()


NEEDS_SCALING = False
NEEDS_RACE = True
