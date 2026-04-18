"""
XGBoost for COMPAS recidivism prediction.
Does not require feature scaling.
"""

from xgboost import XGBClassifier


def build_model():
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # will be overridden in train.py with actual ratio
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
    )


NEEDS_SCALING = False
