"""
Compute per-race FPR, FNR, and PPV gaps for every model on the COMPAS test set.
African-American minus Caucasian.
"""

import importlib
import numpy as np

from data import load_and_preprocess, prepare_features, split_data
from data import scale_features
from runner import MODEL_REGISTRY, MODEL_NAMES


def rate_gap(y_true, y_pred, race, kind):
    def metric(mask):
        yt, yp = y_true[mask], y_pred[mask]
        if kind == 'fpr':
            denom = (yt == 0).sum()
            return ((yt == 0) & (yp == 1)).sum() / denom if denom else float('nan')
        if kind == 'fnr':
            denom = (yt == 1).sum()
            return ((yt == 1) & (yp == 0)).sum() / denom if denom else float('nan')
        if kind == 'ppv':
            denom = (yp == 1).sum()
            return ((yt == 1) & (yp == 1)).sum() / denom if denom else float('nan')
    return metric(race == 'African-American') - metric(race == 'Caucasian')


def evaluate(key, X_train, X_test, y_train, y_test, race_train, race_test):
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
        y_pred = model.predict(X_te, sensitive_features=race_test)
    else:
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

    race_te = np.asarray(race_test)
    yt = y_test
    if selective:
        keep = model.last_keep_mask_
        yt, y_pred, race_te = yt[keep], y_pred[keep], race_te[keep]

    return {
        'fpr_gap': rate_gap(yt, y_pred, race_te, 'fpr'),
        'fnr_gap': rate_gap(yt, y_pred, race_te, 'fnr'),
        'ppv_gap': rate_gap(yt, y_pred, race_te, 'ppv'),
    }


def main():
    df = load_and_preprocess()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test, idx_train, idx_test = split_data(df, X.values, y)
    race_train = df.loc[idx_train, 'race'].values
    race_test = df.loc[idx_test, 'race'].values

    # Also include the COMPAS baseline (decile_score >= 5 on the test set)
    decile_test = df.loc[idx_test, 'decile_score'].values
    compas_pred = (decile_test >= 5).astype(int)
    compas_gaps = {
        'fpr_gap': rate_gap(y_test, compas_pred, race_test, 'fpr'),
        'fnr_gap': rate_gap(y_test, compas_pred, race_test, 'fnr'),
        'ppv_gap': rate_gap(y_test, compas_pred, race_test, 'ppv'),
    }

    rows = [('COMPAS (>=5)', compas_gaps)]
    for key in MODEL_REGISTRY.keys():
        if key == 'imauc':
            continue
        print(f"  {MODEL_NAMES[key]}...")
        gaps = evaluate(key, X_train, X_test, y_train, y_test, race_train, race_test)
        rows.append((MODEL_NAMES[key], gaps))

    print(f"\n{'Model':24s} {'FPRgap':>9s} {'FNRgap':>9s} {'PPVgap':>9s}")
    print('-' * 56)
    for name, g in rows:
        print(f"{name:24s} {g['fpr_gap']:+9.4f} {g['fnr_gap']:+9.4f} {g['ppv_gap']:+9.4f}")


if __name__ == '__main__':
    main()
