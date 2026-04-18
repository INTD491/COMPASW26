"""
Shared data loading and preprocessing for COMPAS recidivism prediction models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(filepath=None):
    """
    Load and preprocess COMPAS dataset following ProPublica's filtering criteria.
    """
    if filepath is None:
        filepath = "../datasets/compas-analysis/compas-scores-two-years.csv"

    df = pd.read_csv(filepath)

    # ProPublica filters
    df = df[df['days_b_screening_arrest'] <= 30]
    df = df[df['days_b_screening_arrest'] >= -30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']

    return df


def prepare_features(df, include_race=False):
    """
    Prepare feature matrix and target from the preprocessed dataframe.

    Returns:
        X: pd.DataFrame of features
        y: np.ndarray of labels
    """
    y = df['two_year_recid'].values

    features = {}

    # Continuous features
    features['age'] = df['age'].values
    features['priors_count'] = df['priors_count'].values
    features['juv_fel_count'] = df['juv_fel_count'].values
    features['juv_misd_count'] = df['juv_misd_count'].values
    features['juv_other_count'] = df['juv_other_count'].values

    # Binary features
    features['sex_male'] = (df['sex'] == 'Male').astype(int).values
    features['charge_degree_felony'] = (df['c_charge_degree'] == 'F').astype(int).values

    # Age category dummies
    age_cat_dummies = pd.get_dummies(df['age_cat'], prefix='age_cat')
    for col in age_cat_dummies.columns:
        features[col] = age_cat_dummies[col].values

    if include_race:
        race_dummies = pd.get_dummies(df['race'], prefix='race')
        for col in race_dummies.columns:
            features[col] = race_dummies[col].values

    X = pd.DataFrame(features)
    return X, y


def split_data(df, X, y, test_size=0.2, random_state=42):
    """
    Split into train/test, returning indices for fairness analysis.

    Returns:
        X_train, X_test, y_train, y_test, idx_train, idx_test
    """
    return train_test_split(
        X, y, df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_features(X_train, X_test):
    """
    Standardize features. Use for models that need it (logistic regression, SVM).
    Binary features get scaled too — this is fine; it just shifts/scales 0/1 values
    and doesn't hurt those models.

    Returns:
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def evaluate_fairness(y_true, y_pred, race):
    """
    Evaluate fairness metrics across racial groups.
    """
    races = ['African-American', 'Caucasian', 'Hispanic', 'Other']
    metrics_by_race = {}

    for r in races:
        mask = (race == r)
        if mask.sum() == 0:
            continue

        y_true_r = y_true[mask]
        y_pred_r = y_pred[mask]

        tn = ((y_true_r == 0) & (y_pred_r == 0)).sum()
        fp = ((y_true_r == 0) & (y_pred_r == 1)).sum()
        fn = ((y_true_r == 1) & (y_pred_r == 0)).sum()
        tp = ((y_true_r == 1) & (y_pred_r == 1)).sum()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        selection_rate = (y_pred_r == 1).sum() / len(y_pred_r)

        metrics_by_race[r] = {
            'n': int(mask.sum()),
            'FPR': fpr,
            'FNR': fnr,
            'PPV': ppv,
            'NPV': npv,
            'Selection_Rate': selection_rate,
        }

    # Print table
    print(f"\n{'Race':20s} {'N':>6s} {'FPR':>6s} {'FNR':>6s} {'PPV':>6s} {'NPV':>6s} {'Sel.Rate':>8s}")
    print("-" * 70)
    for r in races:
        if r in metrics_by_race:
            m = metrics_by_race[r]
            print(f"{r:20s} {m['n']:6d} {m['FPR']:6.3f} {m['FNR']:6.3f} "
                  f"{m['PPV']:6.3f} {m['NPV']:6.3f} {m['Selection_Rate']:8.3f}")

    if 'African-American' in metrics_by_race and 'Caucasian' in metrics_by_race:
        aa = metrics_by_race['African-American']
        cau = metrics_by_race['Caucasian']
        print("\nDisparities (African-American vs Caucasian):")
        if cau['FPR'] > 0:
            print(f"  FPR Ratio: {aa['FPR'] / cau['FPR']:.2f}x")
        if cau['PPV'] > 0:
            print(f"  PPV Ratio: {aa['PPV'] / cau['PPV']:.2f}x")
        if cau['Selection_Rate'] > 0:
            print(f"  Selection Rate Ratio: {aa['Selection_Rate'] / cau['Selection_Rate']:.2f}x")

    return metrics_by_race
