"""
Shared model registry, runner, and metrics for all train_* scripts.
"""

import importlib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from data import scale_features


MODEL_REGISTRY = {
    'logreg':       'models.logistic_regression',
    'dt':           'models.decision_tree',
    'rf':           'models.random_forest',
    'xgb':          'models.xgboost_model',
    'svm':          'models.svm',
    'imauc':        'models.imauc_psvm',
    'fairlearn':    'models.fairlearn_eo',
    'threshold':    'models.threshold_adjusted',
    'threshold_lr': 'models.threshold_adjusted_lr',
    'selective_lr': 'models.logistic_regression_selective',
}

MODEL_NAMES = {
    'logreg':       'Logistic Regression',
    'dt':           'Decision Tree',
    'rf':           'Random Forest',
    'xgb':          'XGBoost',
    'svm':          'SVM',
    'imauc':        'ImAUC-PSVM',
    'fairlearn':    'Fairlearn EO',
    'threshold':    'RF Threshold Adj.',
    'threshold_lr': 'LR Threshold Adj.',
    'selective_lr': 'LR Selective (80%)',
}


def fpr_gap(y_true, y_pred, race, group_a='African-American', group_b='Caucasian'):
    """FPR(group_a) - FPR(group_b). Positive = group_a flagged more often when innocent."""
    def _fpr(mask):
        yt, yp = y_true[mask], y_pred[mask]
        neg = (yt == 0).sum()
        if neg == 0:
            return float('nan')
        return ((yt == 0) & (yp == 1)).sum() / neg
    return _fpr(race == group_a) - _fpr(race == group_b)


def run_model(key, X_train, X_test, y_train, y_test, race_train, race_test):
    mod = importlib.import_module(MODEL_REGISTRY[key])
    model = mod.build_model()
    needs_scaling = getattr(mod, 'NEEDS_SCALING', False)
    needs_race = getattr(mod, 'NEEDS_RACE', False)
    selective = getattr(mod, 'SELECTIVE', False)

    if key == 'xgb':
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        model.set_params(scale_pos_weight=neg / pos)

    if needs_scaling:
        X_tr, X_te, _ = scale_features(X_train, X_test)
    else:
        X_tr, X_te = X_train, X_test

    if needs_race:
        model.fit(X_tr, y_train, sensitive_features=race_train)
        y_pred  = model.predict(X_te, sensitive_features=race_test)
        y_proba = model.predict_proba(X_te, sensitive_features=race_test)[:, 1]
    else:
        model.fit(X_tr, y_train)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

    race_test = np.asarray(race_test)
    if selective:
        keep = model.last_keep_mask_
        y_test, y_pred, y_proba, race_test = y_test[keep], y_pred[keep], y_proba[keep], race_test[keep]

    return {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'auc':       roc_auc_score(y_test, y_proba),
        'fpr_gap':   fpr_gap(y_test, y_pred, race_test),
    }
