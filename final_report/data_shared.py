"""
Shared feature loader for COMPAS and NJ datasets.

Provides a common feature set so models can be trained / evaluated cross-dataset.

Shared features:
    age, sex_male, priors_count, age_cat_*  (3 dummies)

Target:
    2-year recidivism. NJ derived from (Recidivism_Arrest_Year1 == 'Yes')
    OR (Recidivism_Arrest_Year2 == 'Yes').

Sensitive attribute (returned separately, not used as a feature):
    race, mapped to COMPAS labels: African-American, Caucasian (NJ only has these two).
"""

import pandas as pd
import numpy as np


SHARED_FEATURE_NAMES = [
    'age',
    'sex_male',
    'priors_count',
    'age_cat_25 - 45',
    'age_cat_Greater than 45',
    'age_cat_Less than 25',
]


_NJ_AGE_MIDPOINT = {
    '18-22': 20, '23-27': 25, '28-32': 30, '33-37': 35,
    '38-42': 40, '43-47': 45, '48 or older': 53,
}

_NJ_RACE_MAP = {
    'BLACK': 'African-American',
    'WHITE': 'Caucasian',
}


def _age_to_cat(age):
    if age < 25:
        return 'Less than 25'
    if age <= 45:
        return '25 - 45'
    return 'Greater than 45'


def _parse_or_more(val):
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s.endswith('or more'):
        return int(s.split()[0])
    return int(s)


def _build_features(age, sex_male, priors_count):
    out = pd.DataFrame({
        'age': age,
        'sex_male': sex_male,
        'priors_count': priors_count,
    })
    cats = pd.Series(age).apply(_age_to_cat)
    for col in ['age_cat_25 - 45', 'age_cat_Greater than 45', 'age_cat_Less than 25']:
        out[col] = (cats == col.replace('age_cat_', '')).astype(int).values
    return out[SHARED_FEATURE_NAMES]


def load_compas_shared(filepath="../datasets/compas-analysis/compas-scores-two-years.csv"):
    df = pd.read_csv(filepath)
    df = df[df['days_b_screening_arrest'].between(-30, 30)]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']

    X = _build_features(
        age=df['age'].values,
        sex_male=(df['sex'] == 'Male').astype(int).values,
        priors_count=df['priors_count'].values,
    )
    y = df['two_year_recid'].values
    race = df['race'].values
    return X, y, race


def load_nj_shared(filepath="../datasets/compas-analysis/nj_dataset.csv"):
    df = pd.read_csv(filepath)
    df = df[df['Age_at_Release'].isin(_NJ_AGE_MIDPOINT.keys())].copy()
    df = df[df['Race'].isin(_NJ_RACE_MAP.keys())].copy()

    age = df['Age_at_Release'].map(_NJ_AGE_MIDPOINT).values
    priors = (
        df['Prior_Arrest_Episodes_Felony'].apply(_parse_or_more).values
        + df['Prior_Arrest_Episodes_Misd'].apply(_parse_or_more).values
    )
    X = _build_features(
        age=age,
        sex_male=(df['Gender'] == 'M').astype(int).values,
        priors_count=priors,
    )
    y = ((df['Recidivism_Arrest_Year1'] == 'Yes') | (df['Recidivism_Arrest_Year2'] == 'Yes')).astype(int).values
    race = df['Race'].map(_NJ_RACE_MAP).values
    return X, y, race


if __name__ == '__main__':
    X_c, y_c, r_c = load_compas_shared()
    X_n, y_n, r_n = load_nj_shared()
    print(f"COMPAS: n={len(X_c)}  recid_rate={y_c.mean():.1%}")
    print(f"  race counts: {pd.Series(r_c).value_counts().to_dict()}")
    print(f"NJ:     n={len(X_n)}  recid_rate={y_n.mean():.1%}")
    print(f"  race counts: {pd.Series(r_n).value_counts().to_dict()}")
    print(f"\nFeatures: {SHARED_FEATURE_NAMES}")
