import pandas as pd
from sklearn.model_selection import train_test_split

# CLEAN and SPLIT
# print("Passed")
df1 = pd.read_csv("datasets/compas-analysis/compas-scores-two-years-violent.csv")

# violent and non-violent combined
df = pd.read_csv("datasets/compas-analysis/compas-scores-two-years.csv")
# 1. Initial Cleaning (Example for COMPAS)

cleaned_df = df[["age", "c_charge_degree", "race", "age_cat", "score_text", "sex", "priors_count", "days_b_screening_arrest", "decile_score", "is_recid", "two_year_recid", "c_jail_in", "c_jail_out"]]

cleaned_df= cleaned_df[cleaned_df["days_b_screening_arrest"] <= 30 ]
cleaned_df= cleaned_df[cleaned_df["days_b_screening_arrest"] >= -30]

# Missing COMPAS Cases 
cleaned_df = cleaned_df[cleaned_df["is_recid"]!= -1]

# Ordinary Traffic Offenses 
cleaned_df = cleaned_df[cleaned_df["c_charge_degree" ] != "O"]

# Missing score_text
cleaned_df = cleaned_df[cleaned_df["score_text"] != 'N/A']


# 2. Encoding
cleaned_df = pd.get_dummies(cleaned_df, columns=['race', 'sex', 'c_charge_degree'], drop_first=True)

# 3. Splitting: 80% Train, 10% Val, 10% Test
# First, separate 20% for Val+Test
train_df, temp_df = train_test_split(cleaned_df, test_size=0.20, random_state=42, stratify=cleaned_df['two_year_recid'])

# Then split the 20% into two 10% halves
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['two_year_recid'])

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# Data split
train_df.to_csv('datasets/compas-analysis/split/compas-scores-two-years-train.csv', index=False)

val_df.to_csv('datasets/compas-analysis/split/compas-scores-two-years-validation.csv', index=False)

test_df.to_csv('datasets/compas-analysis/split/compas-scores-two-years-test.csv', index=False)
