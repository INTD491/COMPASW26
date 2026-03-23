# --- Setup & Dependencies ---
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import glm

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load raw data
# "./datasets/compas-analysis/compas-scores-two-years.csv"


def clean_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    # print(f"Raw dataset shape: {df.shape}")

    # Select relevant columns
    cleaned_df = df[["age", "c_charge_degree", "race", "age_cat", "score_text", "sex",
                    "priors_count", "days_b_screening_arrest", "decile_score",
                    "is_recid", "two_year_recid", "c_jail_in", "c_jail_out"]].copy()

    # FILTER 1: Charge date within +/-30 days of screening
    cleaned_df = cleaned_df[(cleaned_df["days_b_screening_arrest"] <= 30) &
                            (cleaned_df["days_b_screening_arrest"] >= -30)]
    # print(f"After +/-30 day filter: {len(cleaned_df)}")

    # FILTER 2: Remove missing COMPAS cases (is_recid == -1)
    cleaned_df = cleaned_df[cleaned_df["is_recid"] != -1]
    # print(f"After removing missing cases: {len(cleaned_df)}")

    # FILTER 3: Remove ordinary traffic offenses
    cleaned_df = cleaned_df[cleaned_df["c_charge_degree"] != "O"]
    # print(f"After removing traffic offenses: {len(cleaned_df)}")

    # FILTER 4: Remove N/A scores
    cleaned_df = cleaned_df[cleaned_df["score_text"] != 'N/A']
    # print(f"Final cleaned dataset: {len(cleaned_df)} rows")
    return cleaned_df


def output_graph(csv_path: str):
    cleaned_df = clean_dataset(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    # --- Race Distribution ---
    race_counts = cleaned_df["race"].value_counts()
    colors_race = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#607D8B']
    bars = axes[0].barh(race_counts.index, race_counts.values, color=colors_race, edgecolor='white')
    axes[0].set_title("Race Distribution", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Count")
    for bar, val in zip(bars, race_counts.values):
        pct = val / len(cleaned_df) * 100
        axes[0].text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                    f'{val} ({pct:.1f}%)', va='center', fontsize=9)

    # --- Sex Distribution ---
    sex_counts = cleaned_df["sex"].value_counts()
    axes[1].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%',
                colors=['#42A5F5', '#EF5350'], startangle=90, textprops={'fontsize': 12})
    axes[1].set_title("Sex Distribution", fontsize=14, fontweight='bold')

    # --- Age Category Distribution ---
    age_counts = cleaned_df["age_cat"].value_counts()
    axes[2].bar(age_counts.index, age_counts.values, color=['#66BB6A', '#42A5F5', '#FFA726'], edgecolor='white')
    axes[2].set_title("Age Category Distribution", fontsize=14, fontweight='bold')
    axes[2].set_ylabel("Count")
    for i, (val, idx) in enumerate(zip(age_counts.values, age_counts.index)):
        axes[2].text(i, val + 40, str(val), ha='center', fontsize=10)

    plt.tight_layout()
    # plt.show()

    # print("\n--- Score Text Distribution ---")
    # print(cleaned_df["score_text"].value_counts())
    # print(f"\n--- Race x Sex Crosstab ---")
    # print(pd.crosstab(cleaned_df["race"], cleaned_df["sex"]))
    return fig
