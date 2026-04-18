"""
Exploratory Data Analysis — COMPAS Two-Year Recidivism Dataset

Generates publication-ready plots in final_report/plots/.
Run:  python eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from data import load_and_preprocess

# ---------------------------------------------------------------------------
# Style configuration — clean, corporate aesthetic
# ---------------------------------------------------------------------------
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Muted corporate palette
PALETTE = {
    "primary":   "#2C3E50",
    "secondary": "#7F8C8D",
    "accent":    "#2980B9",
    "highlight": "#E74C3C",
    "green":     "#27AE60",
    "orange":    "#E67E22",
    "purple":    "#8E44AD",
    "light_bg":  "#F8F9FA",
}

RACE_PALETTE = {
    "African-American": "#2980B9",
    "Caucasian":        "#E67E22",
    "Hispanic":         "#27AE60",
    "Other":            "#8E44AD",
    "Asian":            "#7F8C8D",
    "Native American":  "#C0392B",
}

RACE_ORDER = ["African-American", "Caucasian", "Hispanic", "Other", "Asian", "Native American"]

def apply_style():
    """Set global matplotlib/seaborn styling."""
    plt.rcParams.update({
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
        "axes.edgecolor":      "#CCCCCC",
        "axes.grid":           True,
        "grid.color":          "#EEEEEE",
        "grid.linewidth":      0.6,
        "axes.titlesize":      13,
        "axes.titleweight":    "bold",
        "axes.titlepad":       12,
        "axes.labelsize":      11,
        "axes.labelcolor":     "#333333",
        "xtick.color":         "#555555",
        "ytick.color":         "#555555",
        "xtick.labelsize":     9.5,
        "ytick.labelsize":     9.5,
        "legend.fontsize":     9.5,
        "legend.frameon":      False,
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
        "figure.dpi":          150,
        "savefig.dpi":         200,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.2,
    })
    sns.set_style("whitegrid", {
        "axes.edgecolor": "#CCCCCC",
        "grid.color":     "#EEEEEE",
    })

apply_style()


def save(fig, name):
    fig.savefig(PLOT_DIR / f"{name}.png")
    plt.close(fig)
    print(f"  -> plots/{name}.png")


def pct_fmt(x, _):
    return f"{x:.0f}%"


# ===================================================================
#  1. Dataset overview — filtering & sample counts
# ===================================================================
def plot_dataset_overview(df_raw, df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: sample counts before/after filtering
    labels = ["Raw dataset", "After ProPublica filters"]
    counts = [len(df_raw), len(df)]
    bars = axes[0].barh(labels, counts, color=[PALETTE["secondary"], PALETTE["accent"]], height=0.5)
    axes[0].set_xlabel("Number of records")
    axes[0].set_title("Dataset Size Before & After Filtering")
    for bar, c in zip(bars, counts):
        axes[0].text(bar.get_width() - 200, bar.get_y() + bar.get_height() / 2,
                     f"n = {c:,}", va="center", ha="right", fontsize=10,
                     fontweight="bold", color="white")
    axes[0].set_xlim(0, max(counts) * 1.05)

    # Right: missing-value counts for key columns
    key_cols = ["age", "sex", "race", "priors_count", "c_charge_degree",
                "decile_score", "two_year_recid", "days_b_screening_arrest"]
    missing = df_raw[key_cols].isnull().sum().sort_values(ascending=True)
    colors = [PALETTE["highlight"] if v > 0 else PALETTE["green"] for v in missing.values]
    axes[1].barh(missing.index, missing.values, color=colors, height=0.55)
    axes[1].set_xlabel("Missing values (raw dataset)")
    axes[1].set_title("Missing Values in Key Columns")
    for i, v in enumerate(missing.values):
        axes[1].text(v + 5, i, str(v), va="center", fontsize=9, color="#333333")

    fig.suptitle("1  Dataset Overview", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "01_dataset_overview")


# ===================================================================
#  2. Target variable distribution
# ===================================================================
def plot_target(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Pie
    counts = df["two_year_recid"].value_counts().sort_index()
    labels = ["No recidivism", "Recidivism"]
    colors = [PALETTE["accent"], PALETTE["highlight"]]
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_color("white")
    axes[0].set_title("Two-Year Recidivism Rate")

    # Bar with counts
    bars = axes[1].bar(labels, counts, color=colors, width=0.5, edgecolor="white")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Recidivism Counts")
    for bar, c in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                     f"n = {c:,}", ha="center", fontsize=10, fontweight="bold",
                     color="#333333")

    fig.suptitle("2  Target Variable — Two-Year Recidivism", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "02_target_distribution")


# ===================================================================
#  3. Demographic distributions
# ===================================================================
def plot_demographics(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Race
    race_counts = df["race"].value_counts().reindex(RACE_ORDER).dropna()
    colors_r = [RACE_PALETTE[r] for r in race_counts.index]
    bars = axes[0].barh(race_counts.index, race_counts.values, color=colors_r, height=0.6)
    axes[0].set_xlabel("Count")
    axes[0].set_title("Race Distribution")
    axes[0].invert_yaxis()
    for bar, c in zip(bars, race_counts.values):
        axes[0].text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                     f"{c:,} ({c / len(df) * 100:.1f}%)", va="center", fontsize=9)

    # Sex
    sex_counts = df["sex"].value_counts()
    colors_s = [PALETTE["accent"], PALETTE["orange"]]
    wedges, texts, autotexts = axes[1].pie(
        sex_counts, labels=sex_counts.index, colors=colors_s, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_color("white")
    axes[1].set_title("Sex Distribution")

    # Age histogram
    axes[2].hist(df["age"], bins=30, color=PALETTE["accent"], edgecolor="white", linewidth=0.5)
    axes[2].axvline(df["age"].median(), color=PALETTE["highlight"], ls="--", lw=1.5,
                    label=f"Median = {df['age'].median():.0f}")
    axes[2].set_xlabel("Age")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Age Distribution")
    axes[2].legend()

    fig.suptitle("3  Demographic Distributions", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "03_demographics")


# ===================================================================
#  4. Recidivism rates by demographic group
# ===================================================================
def plot_recidivism_by_demographics(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # By race
    race_rates = df.groupby("race")["two_year_recid"].mean().reindex(RACE_ORDER).dropna()
    colors_r = [RACE_PALETTE[r] for r in race_rates.index]
    bars = axes[0].bar(range(len(race_rates)), race_rates.values * 100, color=colors_r, width=0.6)
    axes[0].set_xticks(range(len(race_rates)))
    axes[0].set_xticklabels(race_rates.index, rotation=35, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Recidivism rate (%)")
    axes[0].set_title("By Race")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    for bar, v in zip(bars, race_rates.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                     f"{v * 100:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # By sex
    sex_rates = df.groupby("sex")["two_year_recid"].mean()
    bars = axes[1].bar(sex_rates.index, sex_rates.values * 100,
                       color=[PALETTE["orange"], PALETTE["accent"]], width=0.45)
    axes[1].set_ylabel("Recidivism rate (%)")
    axes[1].set_title("By Sex")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    for bar, v in zip(bars, sex_rates.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v * 100:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # By age category
    age_order = ["Less than 25", "25 - 45", "Greater than 45"]
    age_rates = df.groupby("age_cat")["two_year_recid"].mean().reindex(age_order)
    bars = axes[2].bar(age_order, age_rates.values * 100,
                       color=[PALETTE["highlight"], PALETTE["accent"], PALETTE["green"]], width=0.5)
    axes[2].set_ylabel("Recidivism rate (%)")
    axes[2].set_title("By Age Category")
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    for bar, v in zip(bars, age_rates.values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v * 100:.1f}%", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("4  Recidivism Rates by Demographic Group", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "04_recidivism_by_demographics")


# ===================================================================
#  5. COMPAS decile score distribution
# ===================================================================
def plot_compas_scores(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Overall decile score distribution
    score_counts = df["decile_score"].value_counts().sort_index()
    axes[0].bar(score_counts.index, score_counts.values, color=PALETTE["accent"],
                edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("COMPAS Decile Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Overall Decile Score Distribution")
    axes[0].set_xticks(range(1, 11))

    # Decile score by race (African-American vs Caucasian)
    for race, color in [("African-American", RACE_PALETTE["African-American"]),
                        ("Caucasian", RACE_PALETTE["Caucasian"])]:
        subset = df[df["race"] == race]
        hist_vals = subset["decile_score"].value_counts().sort_index()
        hist_pct = hist_vals / len(subset) * 100
        axes[1].plot(hist_pct.index, hist_pct.values, marker="o", markersize=5,
                     lw=2, label=race, color=color)

    axes[1].set_xlabel("COMPAS Decile Score")
    axes[1].set_ylabel("Proportion of group (%)")
    axes[1].set_title("Score Distribution: African-American vs. Caucasian")
    axes[1].set_xticks(range(1, 11))
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    axes[1].legend()

    fig.suptitle("5  COMPAS Risk Score Distribution", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "05_compas_scores")


# ===================================================================
#  6. COMPAS score vs actual recidivism (calibration)
# ===================================================================
def plot_calibration(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Overall calibration by decile
    cal = df.groupby("decile_score")["two_year_recid"].agg(["mean", "count"])
    axes[0].bar(cal.index, cal["mean"] * 100, color=PALETTE["accent"],
                edgecolor="white", linewidth=0.5)
    axes[0].plot([0.5, 10.5], [0, 100], ls="--", color=PALETTE["secondary"], lw=1,
                 label="Perfect calibration")
    axes[0].set_xlabel("COMPAS Decile Score")
    axes[0].set_ylabel("Actual recidivism rate (%)")
    axes[0].set_title("Calibration: Score vs. Actual Recidivism")
    axes[0].set_xticks(range(1, 11))
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    axes[0].legend()

    # Calibration by race
    for race, color in [("African-American", RACE_PALETTE["African-American"]),
                        ("Caucasian", RACE_PALETTE["Caucasian"])]:
        subset = df[df["race"] == race]
        cal_r = subset.groupby("decile_score")["two_year_recid"].mean()
        axes[1].plot(cal_r.index, cal_r.values * 100, marker="o", markersize=5,
                     lw=2, label=race, color=color)

    axes[1].plot([0.5, 10.5], [0, 100], ls="--", color=PALETTE["secondary"], lw=1,
                 alpha=0.5)
    axes[1].set_xlabel("COMPAS Decile Score")
    axes[1].set_ylabel("Actual recidivism rate (%)")
    axes[1].set_title("Calibration by Race")
    axes[1].set_xticks(range(1, 11))
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    axes[1].legend()

    fig.suptitle("6  COMPAS Score Calibration", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "06_calibration")


# ===================================================================
#  7. COMPAS racial bias — FPR / FNR analysis
# ===================================================================
def plot_compas_fairness(df):
    # Binarize COMPAS: score_text in {Medium, High} → predicted positive
    df = df.copy()
    df["compas_pred"] = (df["score_text"].isin(["Medium", "High"])).astype(int)

    races = ["African-American", "Caucasian", "Hispanic", "Other"]
    metrics = {"FPR": [], "FNR": [], "PPV": [], "Selection Rate": []}
    race_labels = []

    for r in races:
        sub = df[df["race"] == r]
        if len(sub) < 30:
            continue
        race_labels.append(r)
        y_true = sub["two_year_recid"].values
        y_pred = sub["compas_pred"].values

        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()

        metrics["FPR"].append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        metrics["FNR"].append(fn / (fn + tp) if (fn + tp) > 0 else 0)
        metrics["PPV"].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        metrics["Selection Rate"].append(y_pred.mean())

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    metric_names = ["FPR", "FNR", "PPV", "Selection Rate"]
    titles = [
        "False Positive Rate\n(wrongly flagged as high-risk)",
        "False Negative Rate\n(missed actual recidivists)",
        "Positive Predictive Value\n(precision of high-risk label)",
        "Selection Rate\n(proportion flagged high-risk)",
    ]

    for ax, metric, title in zip(axes, metric_names, titles):
        vals = metrics[metric]
        colors = [RACE_PALETTE.get(r, PALETTE["secondary"]) for r in race_labels]
        bars = ax.bar(range(len(race_labels)), [v * 100 for v in vals],
                      color=colors, width=0.55, edgecolor="white")
        ax.set_xticks(range(len(race_labels)))
        ax.set_xticklabels(race_labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v * 100:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    fig.suptitle("7  COMPAS Fairness Metrics by Race", fontsize=15, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "07_compas_fairness")


# ===================================================================
#  8. Prior convictions analysis
# ===================================================================
def plot_priors(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Distribution (clipped at 20 for readability)
    priors_clipped = df["priors_count"].clip(upper=20)
    axes[0].hist(priors_clipped, bins=21, range=(-0.5, 20.5), color=PALETTE["accent"],
                 edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Prior convictions (capped at 20)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Prior Convictions")
    axes[0].text(0.95, 0.95, f"Median: {df['priors_count'].median():.0f}\n"
                 f"Mean: {df['priors_count'].mean():.1f}\n"
                 f"Max: {df['priors_count'].max()}",
                 transform=axes[0].transAxes, va="top", ha="right", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["light_bg"],
                           edgecolor="#CCCCCC"))

    # Recidivism rate by prior count (binned)
    bins = [0, 0.5, 1.5, 3.5, 6.5, 10.5, 50]
    labels_b = ["0", "1", "2-3", "4-6", "7-10", "11+"]
    df_temp = df.copy()
    df_temp["prior_bin"] = pd.cut(df_temp["priors_count"], bins=bins, labels=labels_b)
    rate_by_bin = df_temp.groupby("prior_bin", observed=True)["two_year_recid"].mean() * 100
    rate_by_bin = rate_by_bin.dropna()
    bars = axes[1].bar(rate_by_bin.index.astype(str), rate_by_bin.values,
                       color=PALETTE["accent"], edgecolor="white", width=0.55)
    axes[1].set_xlabel("Prior convictions")
    axes[1].set_ylabel("Recidivism rate (%)")
    axes[1].set_title("Recidivism Rate by Prior Convictions")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    for bar, v in zip(bars, rate_by_bin.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                     f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Priors by race (box plot)
    plot_races = ["African-American", "Caucasian", "Hispanic", "Other"]
    data_box = [df[df["race"] == r]["priors_count"].values for r in plot_races]
    bp = axes[2].boxplot(data_box, tick_labels=plot_races, patch_artist=True,
                         medianprops=dict(color="white", lw=2),
                         flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, r in zip(bp["boxes"], plot_races):
        patch.set_facecolor(RACE_PALETTE[r])
        patch.set_alpha(0.8)
    axes[2].set_ylabel("Prior convictions")
    axes[2].set_title("Priors Distribution by Race")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle("8  Prior Criminal History", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "08_priors")


# ===================================================================
#  9. Age vs recidivism deep-dive
# ===================================================================
def plot_age_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Recidivism rate by age (smoothed)
    age_rates = df.groupby("age")["two_year_recid"].agg(["mean", "count"])
    age_rates = age_rates[age_rates["count"] >= 10]  # filter sparse ages
    axes[0].plot(age_rates.index, age_rates["mean"] * 100, color=PALETTE["accent"], lw=2)
    axes[0].fill_between(age_rates.index, age_rates["mean"] * 100, alpha=0.15,
                         color=PALETTE["accent"])
    axes[0].set_xlabel("Age at screening")
    axes[0].set_ylabel("Recidivism rate (%)")
    axes[0].set_title("Recidivism Rate by Age")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))

    # Age distribution by recidivism outcome
    axes[1].hist(df[df["two_year_recid"] == 0]["age"], bins=30, alpha=0.7,
                 color=PALETTE["accent"], edgecolor="white", linewidth=0.5,
                 label="No recidivism", density=True)
    axes[1].hist(df[df["two_year_recid"] == 1]["age"], bins=30, alpha=0.7,
                 color=PALETTE["highlight"], edgecolor="white", linewidth=0.5,
                 label="Recidivism", density=True)
    axes[1].set_xlabel("Age at screening")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Age Distribution by Outcome")
    axes[1].legend()

    fig.suptitle("9  Age & Recidivism", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "09_age_analysis")


# ===================================================================
# 10. Feature correlation heatmap
# ===================================================================
def plot_correlations(df):
    cols = ["age", "priors_count", "juv_fel_count", "juv_misd_count",
            "juv_other_count", "decile_score", "two_year_recid"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", linewidths=0.8, linecolor="white",
                square=True, ax=ax,
                annot_kws={"size": 9, "fontweight": "bold"},
                cbar_kws={"shrink": 0.75, "label": "Pearson r"})
    ax.set_title("10  Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout()
    save(fig, "10_correlations")


# ===================================================================
# 11. Charge degree analysis
# ===================================================================
def plot_charge_degree(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Counts
    degree_counts = df["c_charge_degree"].value_counts()
    labels_d = ["Felony (F)", "Misdemeanor (M)"]
    colors_d = [PALETTE["highlight"], PALETTE["accent"]]
    bars = axes[0].bar(labels_d, degree_counts.values, color=colors_d, width=0.45,
                       edgecolor="white")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Charge Degree Distribution")
    for bar, c in zip(bars, degree_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                     f"n = {c:,}", ha="center", fontsize=10, fontweight="bold")

    # Recidivism rate by charge degree
    rate_by_degree = df.groupby("c_charge_degree")["two_year_recid"].mean() * 100
    bars = axes[1].bar(labels_d, [rate_by_degree.get("F", 0), rate_by_degree.get("M", 0)],
                       color=colors_d, width=0.45, edgecolor="white")
    axes[1].set_ylabel("Recidivism rate (%)")
    axes[1].set_title("Recidivism Rate by Charge Degree")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("11  Charge Degree Analysis", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "11_charge_degree")


# ===================================================================
# 12. Juvenile history analysis
# ===================================================================
def plot_juvenile(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Proportion with any juvenile record
    df_temp = df.copy()
    df_temp["has_juv"] = ((df_temp["juv_fel_count"] + df_temp["juv_misd_count"]
                           + df_temp["juv_other_count"]) > 0).astype(int)

    juv_counts = df_temp["has_juv"].value_counts().sort_index()
    labels_j = ["No juvenile record", "Has juvenile record"]
    colors_j = [PALETTE["green"], PALETTE["highlight"]]
    wedges, texts, autotexts = axes[0].pie(
        juv_counts.values, labels=labels_j, colors=colors_j, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_color("white")
    axes[0].set_title("Juvenile Record Prevalence")

    # Recidivism rate: with vs without juvenile record
    rate_juv = df_temp.groupby("has_juv")["two_year_recid"].mean() * 100
    bars = axes[1].bar(labels_j, rate_juv.values, color=colors_j, width=0.45,
                       edgecolor="white")
    axes[1].set_ylabel("Recidivism rate (%)")
    axes[1].set_title("Recidivism Rate by Juvenile Record")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("12  Juvenile Criminal History", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "12_juvenile")


# ===================================================================
# Main
# ===================================================================
def main():
    print("Loading data...")
    df_raw = pd.read_csv("../datasets/compas-analysis/compas-scores-two-years.csv")
    df = load_and_preprocess()

    print(f"Raw: {len(df_raw):,} records | Filtered: {len(df):,} records")
    print(f"Recidivism rate: {df['two_year_recid'].mean():.1%}\n")
    print("Generating plots...\n")

    plot_dataset_overview(df_raw, df)
    plot_target(df)
    plot_demographics(df)
    plot_recidivism_by_demographics(df)
    plot_compas_scores(df)
    plot_calibration(df)
    plot_compas_fairness(df)
    plot_priors(df)
    plot_age_analysis(df)
    plot_correlations(df)
    plot_charge_degree(df)
    plot_juvenile(df)

    print(f"\nDone — {12} plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
