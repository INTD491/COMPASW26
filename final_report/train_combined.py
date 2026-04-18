"""
Combined-training evaluation: pool COMPAS + NJ training splits, test on each
test split separately. Asks: does training on more (mixed) data help fairness
or accuracy on either dataset?

Usage:
    python train_combined.py logreg
    python train_combined.py all
"""

import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from data_shared import load_compas_shared, load_nj_shared
from runner import MODEL_REGISTRY, MODEL_NAMES, run_model


def split(X, y, race, seed=42):
    return train_test_split(X.values, y, race, test_size=0.2, random_state=seed, stratify=y)


def main():
    parser = argparse.ArgumentParser(description="Pooled COMPAS+NJ training, test on each separately")
    parser.add_argument('models', nargs='+', choices=list(MODEL_REGISTRY.keys()) + ['all'])
    args = parser.parse_args()
    keys = list(MODEL_REGISTRY.keys()) if 'all' in args.models else args.models

    print("Loading datasets...")
    Xc, yc, rc = load_compas_shared()
    Xn, yn, rn = load_nj_shared()
    cXtr, cXte, cytr, cyte, crtr, crte = split(Xc, yc, rc)
    nXtr, nXte, nytr, nyte, nrtr, nrte = split(Xn, yn, rn)

    Xtr_pool = np.vstack([cXtr, nXtr])
    ytr_pool = np.concatenate([cytr, nytr])
    rtr_pool = np.concatenate([crtr, nrtr])

    print(f"Pooled train: {len(ytr_pool)} | Test on COMPAS: {len(cyte)} | Test on NJ: {len(nyte)}")

    conditions = [
        ('Pooled->COMPAS', Xtr_pool, cXte, ytr_pool, cyte, rtr_pool, crte),
        ('Pooled->NJ',     Xtr_pool, nXte, ytr_pool, nyte, rtr_pool, nrte),
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
