"""
Cross-dataset evaluation: train on one dataset, test on the other.

For each model, runs 4 conditions:
    COMPAS -> COMPAS  (in-distribution baseline)
    COMPAS -> NJ      (transfer)
    NJ     -> NJ      (in-distribution baseline)
    NJ     -> COMPAS  (reverse transfer)

Uses the shared feature set from data_shared.py so the same model trained on
either dataset can predict on the other.

Usage:
    python train_cross.py logreg
    python train_cross.py all
"""

import argparse
from sklearn.model_selection import train_test_split

from data_shared import load_compas_shared, load_nj_shared
from runner import MODEL_REGISTRY, MODEL_NAMES, run_model


def split(X, y, race, seed=42):
    return train_test_split(X.values, y, race, test_size=0.2, random_state=seed, stratify=y)


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset training/eval (COMPAS <-> NJ)")
    parser.add_argument('models', nargs='+', choices=list(MODEL_REGISTRY.keys()) + ['all'])
    args = parser.parse_args()
    keys = list(MODEL_REGISTRY.keys()) if 'all' in args.models else args.models

    print("Loading datasets...")
    Xc, yc, rc = load_compas_shared()
    Xn, yn, rn = load_nj_shared()
    cXtr, cXte, cytr, cyte, crtr, crte = split(Xc, yc, rc)
    nXtr, nXte, nytr, nyte, nrtr, nrte = split(Xn, yn, rn)
    print(f"COMPAS: train={len(cytr)} test={len(cyte)} (recid {yc.mean():.1%})")
    print(f"NJ:     train={len(nytr)} test={len(nyte)} (recid {yn.mean():.1%})")

    conditions = [
        ('COMPAS->COMPAS', cXtr, cXte, cytr, cyte, crtr, crte),
        ('COMPAS->NJ',     cXtr, nXte, cytr, nyte, crtr, nrte),
        ('NJ->NJ',         nXtr, nXte, nytr, nyte, nrtr, nrte),
        ('NJ->COMPAS',     nXtr, cXte, nytr, cyte, nrtr, crte),
    ]

    results = {}
    for key in keys:
        results[key] = {}
        for label, Xtr, Xte, ytr, yte, rtr, rte in conditions:
            print(f"  {MODEL_NAMES[key]} ({label})...")
            results[key][label] = run_model(key, Xtr, Xte, ytr, yte, rtr, rte)

    print(f"\n{'Model':22s} {'Train->Test':>16s} {'Acc':>7s} {'AUC':>7s} {'FPRgap':>9s}")
    print("-" * 66)
    for key in keys:
        for label in [c[0] for c in conditions]:
            r = results[key][label]
            print(f"{MODEL_NAMES[key]:22s} {label:>16s} {r['accuracy']:7.4f} "
                  f"{r['auc']:7.4f} {r['fpr_gap']:+9.4f}")
        print()


if __name__ == '__main__':
    main()
