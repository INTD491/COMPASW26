"""
Research-grade uncertainty-coverage tradeoff plot for selective prediction.

This script trains a logistic regression model on the COMPAS training split,
then evaluates selective prediction at coverage levels from 50% to 100%.
At each coverage, it reports performance and fairness gaps (African-American
minus Caucasian) on the retained subset.

Outputs:
- plots/uncertainty_coverage_tradeoff.png
- plots/uncertainty_coverage_tradeoff.pdf
- plots/uncertainty_coverage_tradeoff_metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import load_and_preprocess, prepare_features, split_data, scale_features


OUT_DIR = Path("plots")
OUT_PNG = OUT_DIR / "uncertainty_coverage_tradeoff.png"
OUT_PDF = OUT_DIR / "uncertainty_coverage_tradeoff.pdf"
OUT_CSV = OUT_DIR / "uncertainty_coverage_tradeoff_metrics.csv"


def _rate_gap(y_true, y_pred, race, kind):
    def metric(mask):
        yt, yp = y_true[mask], y_pred[mask]
        if kind == "fpr":
            denom = (yt == 0).sum()
            return ((yt == 0) & (yp == 1)).sum() / denom if denom else np.nan
        if kind == "fnr":
            denom = (yt == 1).sum()
            return ((yt == 1) & (yp == 0)).sum() / denom if denom else np.nan
        if kind == "ppv":
            denom = (yp == 1).sum()
            return ((yt == 1) & (yp == 1)).sum() / denom if denom else np.nan
        raise ValueError(f"Unknown gap kind: {kind}")

    aa = metric(race == "African-American")
    cau = metric(race == "Caucasian")
    return aa - cau


def _compute_metrics(y_true, y_pred, y_proba, race):
    auc = np.nan
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_proba)

    fpr_gap = _rate_gap(y_true, y_pred, race, "fpr")
    fnr_gap = _rate_gap(y_true, y_pred, race, "fnr")
    ppv_gap = _rate_gap(y_true, y_pred, race, "ppv")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": auc,
        "fpr_gap": fpr_gap,
        "fnr_gap": fnr_gap,
        "ppv_gap": ppv_gap,
        "abs_fpr_gap": np.abs(fpr_gap),
        "abs_fnr_gap": np.abs(fnr_gap),
        "abs_ppv_gap": np.abs(ppv_gap),
    }


def _retained_mask_from_coverage(proba, coverage):
    confidence = np.abs(proba - 0.5)
    n = len(confidence)
    n_keep = max(1, int(round(n * coverage)))
    keep_idx = np.argsort(confidence)[::-1][:n_keep]
    mask = np.zeros(n, dtype=bool)
    mask[keep_idx] = True
    return mask


def _research_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _annotate_at_80(ax, df_metrics, y_col, color):
    row = df_metrics.loc[np.isclose(df_metrics["coverage"], 0.80)].iloc[0]
    x, y = row["coverage_pct"], row[y_col]
    ax.scatter([x], [y], color=color, s=34, zorder=5, edgecolor="white", linewidth=0.7)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _research_style()

    df = load_and_preprocess()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test, idx_train, idx_test = split_data(df, X.values, y)
    X_tr, X_te, _ = scale_features(X_train, X_test)

    race_test = df.loc[idx_test, "race"].values

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
        keep = _retained_mask_from_coverage(proba, cov)
        y_cov = y_test[keep]
        pred_cov = pred[keep]
        proba_cov = proba[keep]
        race_cov = race_test[keep]

        metrics = _compute_metrics(y_cov, pred_cov, proba_cov, race_cov)
        rows.append(
            {
                "coverage": cov,
                "coverage_pct": cov * 100.0,
                "n_retained": int(keep.sum()),
                **metrics,
            }
        )

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(OUT_CSV, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.4), dpi=200, sharex=True)

    perf_colors = {
        "accuracy": "#1b6ca8",
        "f1": "#2f9e44",
        "auc": "#e07a1f",
    }
    fair_colors = {
        "abs_fpr_gap": "#c53d13",
        "abs_fnr_gap": "#6f42c1",
        "abs_ppv_gap": "#0f766e",
    }

    ax = axes[0]
    ax.plot(df_metrics["coverage_pct"], df_metrics["accuracy"], color=perf_colors["accuracy"], marker="o", linewidth=2.0, label="Accuracy")
    ax.plot(df_metrics["coverage_pct"], df_metrics["f1"], color=perf_colors["f1"], marker="s", linewidth=2.0, label="F1")
    ax.plot(df_metrics["coverage_pct"], df_metrics["auc"], color=perf_colors["auc"], marker="^", linewidth=2.0, label="AUC-ROC")

    _annotate_at_80(ax, df_metrics, "accuracy", perf_colors["accuracy"])
    _annotate_at_80(ax, df_metrics, "f1", perf_colors["f1"])
    _annotate_at_80(ax, df_metrics, "auc", perf_colors["auc"])

    ax.axvline(80, color="#4b5563", linestyle="--", linewidth=1.1, alpha=0.75)
    ax.set_ylabel("Performance")
    ax.set_ylim(0.54, 0.79)
    ax.set_title("Selective Prediction Trade-off Across Coverage (COMPAS Test Set)")
    ax.legend(loc="lower right", ncol=3, frameon=False)

    ax = axes[1]
    ax.plot(df_metrics["coverage_pct"], df_metrics["abs_fpr_gap"], color=fair_colors["abs_fpr_gap"], marker="o", linewidth=2.0, label=r"$|\Delta \mathrm{FPR}|$")
    ax.plot(df_metrics["coverage_pct"], df_metrics["abs_fnr_gap"], color=fair_colors["abs_fnr_gap"], marker="s", linewidth=2.0, label=r"$|\Delta \mathrm{FNR}|$")
    ax.plot(df_metrics["coverage_pct"], df_metrics["abs_ppv_gap"], color=fair_colors["abs_ppv_gap"], marker="^", linewidth=2.0, label=r"$|\Delta \mathrm{PPV}|$")

    _annotate_at_80(ax, df_metrics, "abs_fpr_gap", fair_colors["abs_fpr_gap"])
    _annotate_at_80(ax, df_metrics, "abs_fnr_gap", fair_colors["abs_fnr_gap"])
    _annotate_at_80(ax, df_metrics, "abs_ppv_gap", fair_colors["abs_ppv_gap"])

    ax.axvline(80, color="#4b5563", linestyle="--", linewidth=1.1, alpha=0.75)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Fairness Gap Magnitude")
    ax.set_ylim(0.0, 0.45)
    ax.legend(loc="upper right", ncol=3, frameon=False)

    row80 = df_metrics.loc[np.isclose(df_metrics["coverage"], 0.80)].iloc[0]
    subtitle = (
        "Gap definition: African-American minus Caucasian; bottom panel reports absolute gap. "
        f"At 80% coverage: Acc={row80['accuracy']:.3f}, F1={row80['f1']:.3f}, AUC={row80['auc']:.3f}."
    )
    fig.text(0.5, 0.005, subtitle, ha="center", va="bottom", fontsize=9)

    plt.tight_layout(rect=[0.02, 0.04, 0.995, 1.0])
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    print(f"Saved figure (PNG): {OUT_PNG}")
    print(f"Saved figure (PDF): {OUT_PDF}")
    print(f"Saved metrics CSV:  {OUT_CSV}")


if __name__ == "__main__":
    main()
