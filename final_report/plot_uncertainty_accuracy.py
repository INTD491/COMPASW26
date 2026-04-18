"""
Simple accuracy-vs-coverage plot for selective prediction.

Coverage is swept from 50% to 100% (in 5% steps), using logistic regression
on the standard COMPAS split. Metrics are computed on the retained subset.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data import load_and_preprocess, prepare_features, split_data, scale_features


OUT_DIR = Path("plots")
OUT_PNG = OUT_DIR / "uncertainty_accuracy_vs_coverage.png"
OUT_CSV = OUT_DIR / "uncertainty_accuracy_vs_coverage.csv"


def retained_mask(proba, coverage):
    confidence = np.abs(proba - 0.5)
    n_keep = max(1, int(round(len(confidence) * coverage)))
    idx = np.argsort(confidence)[::-1][:n_keep]
    mask = np.zeros(len(confidence), dtype=bool)
    mask[idx] = True
    return mask


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_preprocess()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test, _, _ = split_data(df, X.values, y)
    X_tr, X_te, _ = scale_features(X_train, X_test)

    lr = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    lr.fit(X_tr, y_train)

    proba = lr.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    coverages = np.round(np.arange(0.50, 1.001, 0.05), 2)
    rows = []

    for cov in coverages:
        keep = retained_mask(proba, cov)
        acc = accuracy_score(y_test[keep], pred[keep])
        rows.append(
            {
                "coverage": cov,
                "coverage_pct": cov * 100,
                "n_retained": int(keep.sum()),
                "accuracy": acc,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_CSV, index=False)

    plt.figure(figsize=(7.2, 4.4), dpi=180)
    plt.plot(metrics["coverage_pct"], metrics["accuracy"], marker="o", linewidth=2, color="#1f77b4")
    plt.axvline(80, color="#666666", linestyle="--", linewidth=1)

    row80 = metrics.loc[np.isclose(metrics["coverage"], 0.80)].iloc[0]
    plt.scatter([80], [row80["accuracy"]], color="#1f77b4", s=40, zorder=4)
    plt.text(80.8, row80["accuracy"] + 0.002, f"80%: {row80['accuracy']:.3f}", fontsize=9)

    plt.title("Accuracy vs Coverage (Selective Logistic Regression)")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Accuracy")
    plt.ylim(0.67, 0.78)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_PNG, bbox_inches="tight")

    print(f"Saved plot: {OUT_PNG}")
    print(f"Saved metrics: {OUT_CSV}")


if __name__ == "__main__":
    main()
