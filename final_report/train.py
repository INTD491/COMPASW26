"""
Train COMPAS recidivism models on the COMPAS dataset (rich feature set).

Usage:
    python train.py logreg                  # logistic regression
    python train.py dt                      # decision tree
    python train.py rf                      # random forest
    python train.py xgb                     # xgboost
    python train.py svm                     # support vector machine
    python train.py imauc                   # ImAUC-PSVM (AUC-optimised)
    python train.py fairlearn               # Fairlearn EqualizedOdds (RF base)
    python train.py threshold               # RF + per-group threshold adjustment
    python train.py threshold_lr            # LR + per-group threshold adjustment
    python train.py all                     # run all models
    python train.py logreg rf xgb           # run a subset
"""

import argparse

from data import load_and_preprocess, prepare_features, split_data
from runner import MODEL_REGISTRY, MODEL_NAMES, run_model


def main():
    parser = argparse.ArgumentParser(description="Train COMPAS recidivism models (rich features)")
    parser.add_argument(
        'models',
        nargs='+',
        choices=list(MODEL_REGISTRY.keys()) + ['all'],
        help="Which model(s) to train",
    )
    parser.add_argument('--include-race', action='store_true', help="Include race as a feature")
    args = parser.parse_args()

    keys = list(MODEL_REGISTRY.keys()) if 'all' in args.models else args.models

    df = load_and_preprocess()
    X, y = prepare_features(df, include_race=args.include_race)
    X_train, X_test, y_train, y_test, idx_train, idx_test = split_data(df, X.values, y)
    race_train = df.loc[idx_train, 'race'].values
    race_test  = df.loc[idx_test,  'race'].values

    print(f"Samples: {len(df)} | Recidivism rate: {y.mean():.1%}")
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")

    results = {}
    for key in keys:
        print(f"  Training {MODEL_NAMES[key]}...")
        results[key] = run_model(key, X_train, X_test, y_train, y_test, race_train, race_test)

    print(f"\n{'Model':22s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s} {'FPRgap':>9s}")
    print("-" * 70)
    for key in keys:
        r = results[key]
        print(f"{MODEL_NAMES[key]:22s} {r['accuracy']:7.4f} {r['precision']:7.4f} "
              f"{r['recall']:7.4f} {r['f1']:7.4f} {r['auc']:7.4f} {r['fpr_gap']:+9.4f}")


if __name__ == '__main__':
    main()
