"""
Random Forest with per-group threshold adjustment.
Fits an RF, then chooses a separate decision threshold per racial group
so that each group's FPR matches the Caucasian baseline.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve


class ThresholdAdjustedRF:
    def __init__(self, target_race='Caucasian'):
        self.target_race = target_race
        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=10,
            min_samples_leaf=4, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        self.thresholds_ = {}
        self.target_fpr_ = None

    def fit(self, X, y, sensitive_features):
        self.rf.fit(X, y)
        proba = self.rf.predict_proba(X)[:, 1]
        sf = np.asarray(sensitive_features)

        target_mask = (sf == self.target_race)
        if target_mask.sum() == 0:
            self.target_fpr_ = 0.3
        else:
            tp = ((proba[target_mask] >= 0.5) & (y[target_mask] == 0))
            neg = (y[target_mask] == 0)
            self.target_fpr_ = tp.sum() / max(neg.sum(), 1)

        for r in np.unique(sf):
            mask = (sf == r)
            if mask.sum() < 10 or (y[mask] == 0).sum() == 0:
                self.thresholds_[r] = 0.5
                continue
            fpr_arr, _, thresh_arr = roc_curve(y[mask], proba[mask])
            idx = int(np.argmin(np.abs(fpr_arr - self.target_fpr_)))
            self.thresholds_[r] = float(thresh_arr[idx])
        return self

    def predict(self, X, sensitive_features=None):
        proba = self.rf.predict_proba(X)[:, 1]
        if sensitive_features is None:
            return (proba >= 0.5).astype(int)
        sf = np.asarray(sensitive_features)
        out = (proba >= 0.5).astype(int)
        for r, t in self.thresholds_.items():
            mask = (sf == r)
            if mask.any():
                out[mask] = (proba[mask] >= t).astype(int)
        return out

    def predict_proba(self, X, sensitive_features=None):
        return self.rf.predict_proba(X)


def build_model():
    return ThresholdAdjustedRF()


NEEDS_SCALING = False
NEEDS_RACE = True
