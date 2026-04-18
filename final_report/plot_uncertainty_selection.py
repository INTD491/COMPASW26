"""
Visualise the selective-prediction abstention rule.

Top:    histogram of predicted probabilities on the COMPAS test set,
        with the central abstain band (20% closest to p=0.5) shaded.
Bottom: distribution of confidence |p-0.5|, with the abstention threshold
        marked.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from data import load_and_preprocess, prepare_features, split_data, scale_features


COVERAGE = 0.80
OUT = 'plots/uncertainty_selection.png'


def main():
    df = load_and_preprocess()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test, _, _ = split_data(df, X.values, y)
    X_tr, X_te, _ = scale_features(X_train, X_test)

    lr = LogisticRegression(
        penalty='l2', C=1.0, solver='liblinear',
        max_iter=1000, class_weight='balanced', random_state=42,
    )
    lr.fit(X_tr, y_train)
    proba = lr.predict_proba(X_te)[:, 1]
    confidence = np.abs(proba - 0.5)

    # cutoff that retains COVERAGE fraction (i.e. drops 1-COVERAGE most uncertain)
    cutoff = np.quantile(confidence, 1 - COVERAGE)
    kept = confidence >= cutoff
    abstained = ~kept
    p_low = 0.5 - cutoff
    p_high = 0.5 + cutoff

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5))

    # --- top: predicted probability histogram with abstain band ---
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    ax.hist(proba[kept], bins=bins, color='#4C72B0', alpha=0.85,
            label=f'Retained ({kept.sum()}, {100*kept.mean():.0f}%)',
            edgecolor='white', linewidth=0.5)
    ax.hist(proba[abstained], bins=bins, color='#C44E52', alpha=0.85,
            label=f'Abstained ({abstained.sum()}, {100*abstained.mean():.0f}%)',
            edgecolor='white', linewidth=0.5)
    ax.axvspan(p_low, p_high, color='gray', alpha=0.12, zorder=0)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Predicted P(recidivate)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of predicted probabilities (COMPAS test set)')
    ax.legend(loc='upper center', frameon=False)
    ax.set_xlim(0, 1)

    # --- bottom: confidence |p - 0.5| with cutoff line ---
    ax = axes[1]
    ax.hist(confidence, bins=40, color='#55A868', alpha=0.85,
            edgecolor='white', linewidth=0.5)
    ax.axvline(cutoff, color='black', linestyle='--', linewidth=1.2,
               label=f'Abstain cutoff = {cutoff:.3f}')
    ax.set_xlabel(r'Confidence $|\hat{p} - 0.5|$')
    ax.set_ylabel('Count')
    ax.set_title(f'Confidence distribution (bottom {int(100*(1-COVERAGE))}% abstained)')
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(OUT, dpi=160, bbox_inches='tight')
    print(f"Saved: {OUT}")
    print(f"Cutoff confidence: {cutoff:.4f}  ->  abstain band: [{p_low:.3f}, {p_high:.3f}]")
    print(f"Retained: {kept.sum()} / {len(kept)} ({100*kept.mean():.1f}%)")


if __name__ == '__main__':
    main()
