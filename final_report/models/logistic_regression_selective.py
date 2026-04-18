"""
Logistic Regression with selective prediction (uncertainty-based abstention).

Trains LR normally, then at predict time abstains on the most uncertain
1-coverage fraction of samples (those with predicted probability closest
to 0.5). Reported metrics are computed on the retained subset only.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class SelectiveLR:
    def __init__(self, coverage=0.80):
        self.coverage = coverage
        self.lr = LogisticRegression(
            penalty='l2', C=1.0, solver='liblinear',
            max_iter=1000, class_weight='balanced', random_state=42,
        )
        self.last_keep_mask_ = None

    def fit(self, X, y):
        self.lr.fit(X, y)
        return self

    def _keep_mask(self, X):
        proba = self.lr.predict_proba(X)[:, 1]
        confidence = np.abs(proba - 0.5)
        n = len(proba)
        n_keep = max(1, int(round(n * self.coverage)))
        keep_idx = np.argsort(confidence)[::-1][:n_keep]
        mask = np.zeros(n, dtype=bool)
        mask[keep_idx] = True
        return mask

    def predict(self, X):
        self.last_keep_mask_ = self._keep_mask(X)
        return (self.lr.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        return self.lr.predict_proba(X)


def build_model():
    return SelectiveLR(coverage=0.80)


NEEDS_SCALING = True
SELECTIVE = True
