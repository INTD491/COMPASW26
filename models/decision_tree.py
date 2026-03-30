"""
Decision Tree Model for COMPAS Two-Year Recidivism Prediction

Key Literature & Citations:
---------------------------
1. Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016).
   "Machine Bias: There's software used across the country to predict future criminals.
   And it's biased against blacks." ProPublica.

2. Dressel, J., & Farid, H. (2018).
   "The accuracy, fairness, and limits of predicting recidivism."
   Science advances, 4(1), eaao5580.
   - Showed that simple models with few features can match COMPAS accuracy

3. Rudin, C., Wang, C., & Coker, B. (2020).
   "The age of secrecy and unfairness in recidivism prediction."
   Harvard Data Science Review, 2(1).
   - Argues for interpretable models; decision trees are fully transparent

4. Tollenaar, N., & Van der Heijden, P. G. (2013).
   "Which method predicts recidivism best?: a comparison of statistical,
   machine learning and data mining predictive models."
   Journal of the Royal Statistical Society, 176(2), 565-584.
   - Compared logistic regression, decision trees, and random forests

5. Lim, T. S., Loh, W. Y., & Shih, Y. S. (2000).
   "A comparison of prediction accuracy, complexity, and training time of
   thirty-three old and new classification algorithms."
   Machine learning, 40(3), 203-228.

Decision Tree Advantages for Criminal Justice:
----------------------------------------------
- Fully interpretable: the exact decision path for any defendant can be traced
- Produces human-readable rules (e.g., "if age < 25 AND priors > 3 → high risk")
- No feature scaling required
- Naturally handles feature interactions
- Can expose proxy variable patterns in a transparent way

Decision Tree Limitations:
--------------------------
- Prone to overfitting without proper depth/leaf constraints
- High variance (small data changes can produce very different trees)
- Generally lower predictive performance than ensemble methods (Random Forest)
- Unlike logistic regression or random forests, a single tree's threshold
  is harder to calibrate for fairness interventions
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV

# Root of the repo regardless of where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath):
    """
    Load and preprocess COMPAS dataset following ProPublica's filtering criteria.
    """
    df = pd.read_csv(filepath)

    # Apply ProPublica filters
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']

    return df


def prepare_features(df, include_race=False):
    """
    Prepare features for the decision tree model.

    Decision trees do not require feature scaling, but we use the same
    feature set as the logistic regression and random forest baselines
    for a fair comparison.

    Args:
        df: DataFrame with COMPAS data
        include_race: Whether to include race as a feature (for fairness analysis)

    Returns:
        X: Feature matrix (numpy array)
        y: Target variable (two_year_recid)
        feature_names: List of feature names
    """
    y = df['two_year_recid'].values

    features = {}

    # Continuous features
    features['age'] = df['age'].values
    features['priors_count'] = df['priors_count'].values

    # Juvenile history (COMPAS dataset only)
    if 'juv_fel_count' in df.columns:
        features['juv_fel_count'] = df['juv_fel_count'].values
    if 'juv_misd_count' in df.columns:
        features['juv_misd_count'] = df['juv_misd_count'].values
    if 'juv_other_count' in df.columns:
        features['juv_other_count'] = df['juv_other_count'].values

    # Binary features
    features['sex_male'] = (df['sex'] == 'Male').astype(int).values
    features['charge_degree_felony'] = (df['c_charge_degree'] == 'F').astype(int).values

    # Age categories
    age_cat_dummies = pd.get_dummies(df['age_cat'], prefix='age_cat')
    for col in age_cat_dummies.columns:
        features[col] = age_cat_dummies[col].values

    # Optional: Race (for fairness-aware models)
    if include_race:
        race_dummies = pd.get_dummies(df['race'], prefix='race')
        for col in race_dummies.columns:
            features[col] = race_dummies[col].values

    X = pd.DataFrame(features)
    feature_names = list(X.columns)

    return X.values, y, feature_names


def train_decision_tree(X_train, y_train, tune_hyperparameters=False):
    """
    Train a decision tree classifier with optional hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        tune_hyperparameters: Whether to perform grid search (slower but better)

    Returns:
        Trained DecisionTreeClassifier
    """
    if tune_hyperparameters:
        param_grid = {
            'max_depth': [3, 5, 8, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', None]
        }

        dt = DecisionTreeClassifier(random_state=42)

        grid_search = GridSearchCV(
            dt,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    else:
        # Default parameters — depth-limited to reduce overfitting
        # max_depth=10 balances interpretability and performance
        dt = DecisionTreeClassifier(
            criterion='gini',          # Gini impurity for splitting
            max_depth=10,              # Limit depth to prevent overfitting
            min_samples_split=10,      # Minimum samples to split a node
            min_samples_leaf=4,        # Minimum samples at leaf node
            class_weight='balanced',   # Handle class imbalance
            random_state=42
        )

        dt.fit(X_train, y_train)

        return dt


def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    """
    Comprehensive evaluation of the decision tree model.
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | ROC-AUC: {test_auc:.4f}")
    print(f"Tree depth: {model.get_depth()} | Leaves: {model.get_n_leaves()}")

    # Feature importance (based on Gini impurity reduction)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 5 Features:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:6.3f}")

    return y_test_pred, y_test_proba, importance_df


def print_tree_rules(model, feature_names, max_depth=3):
    """
    Print the top decision rules of the tree as human-readable text.
    Useful for transparency and interpretability analysis.

    Args:
        model: Trained DecisionTreeClassifier
        feature_names: List of feature names
        max_depth: How many levels of rules to display
    """
    print(f"\n=== Decision Tree Rules (top {max_depth} levels) ===")
    rules = export_text(model, feature_names=feature_names, max_depth=max_depth)
    print(rules)


def plot_feature_importance(importance_df, save_path=None):
    """
    Plot feature importance from the decision tree (Gini-based).
    """
    plt.figure(figsize=(10, 6))

    top_features = importance_df.head(15)

    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance (Gini Impurity Reduction)')
    plt.title('Decision Tree Feature Importance (Top 15)')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    path = save_path if save_path else 'dt_feature_importance.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_fairness(y_true, y_pred, y_proba, race, threshold=0.5):
    """
    Evaluate fairness metrics across racial groups.

    Metrics computed:
    - FPR: False Positive Rate (false alarms)
    - FNR: False Negative Rate (missed recidivists)
    - PPV: Positive Predictive Value (precision)
    - NPV: Negative Predictive Value
    - Selection Rate: Proportion predicted positive
    """
    print("\n=== Fairness Analysis ===")

    races = ['African-American', 'Caucasian', 'Hispanic', 'Other']
    metrics_by_race = {}

    for r in races:
        mask = (race == r)
        if mask.sum() == 0:
            continue

        y_true_r = y_true[mask]
        y_pred_r = y_pred[mask]

        # Confusion matrix components
        tn = ((y_true_r == 0) & (y_pred_r == 0)).sum()
        fp = ((y_true_r == 0) & (y_pred_r == 1)).sum()
        fn = ((y_true_r == 1) & (y_pred_r == 0)).sum()
        tp = ((y_true_r == 1) & (y_pred_r == 1)).sum()

        # Calculate metrics
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        selection_rate = (y_pred_r == 1).sum() / len(y_pred_r)

        metrics_by_race[r] = {
            'n': mask.sum(),
            'FPR': fpr,
            'FNR': fnr,
            'PPV': ppv,
            'NPV': npv,
            'Selection_Rate': selection_rate
        }

    # Print metrics
    print(f"\n{'Race':20s} {'N':>6s} {'FPR':>6s} {'FNR':>6s} {'PPV':>6s} {'NPV':>6s} {'Sel.Rate':>8s}")
    print("-" * 70)

    for r in races:
        if r in metrics_by_race:
            m = metrics_by_race[r]
            print(f"{r:20s} {m['n']:6d} {m['FPR']:6.3f} {m['FNR']:6.3f} {m['PPV']:6.3f} {m['NPV']:6.3f} {m['Selection_Rate']:8.3f}")

    # Calculate disparities
    if 'African-American' in metrics_by_race and 'Caucasian' in metrics_by_race:
        aa = metrics_by_race['African-American']
        cau = metrics_by_race['Caucasian']

        print("\nDisparities (African-American vs Caucasian):")
        print(f"  FPR Ratio: {aa['FPR'] / cau['FPR']:.2f}x" if cau['FPR'] > 0 else "  FPR Ratio: N/A")
        print(f"  PPV Ratio: {aa['PPV'] / cau['PPV']:.2f}x" if cau['PPV'] > 0 else "  PPV Ratio: N/A")
        print(f"  Selection Rate Ratio: {aa['Selection_Rate'] / cau['Selection_Rate']:.2f}x" if cau['Selection_Rate'] > 0 else "  Selection Rate Ratio: N/A")

    return metrics_by_race


def cross_validate_model(X, y, cv_folds=5):
    """
    Perform cross-validation to assess model stability.
    """
    dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    cv_scores = cross_val_score(dt, X, y, cv=cv, scoring='roc_auc')

    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return cv_scores


def main():
    """
    Main execution function.
    """
    print("\n=== Decision Tree Model ===")

    # Load data
    filepath = os.path.join(BASE_DIR, 'datasets', 'compas-analysis', 'compas-scores-two-years.csv')
    df = load_and_preprocess_data(filepath)
    print(f"Samples: {len(df)} | Recidivism rate: {df['two_year_recid'].mean():.1%}")

    # Prepare features (race-blind model)
    X, y, feature_names = prepare_features(df, include_race=False)

    # Split data with indices to track demographics
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    # Train model (set tune_hyperparameters=True for better performance but slower)
    model = train_decision_tree(X_train, y_train, tune_hyperparameters=False)

    # Evaluate model
    y_test_pred, y_test_proba, importance_df = evaluate_model(
        model, X_train, y_train, X_test, y_test, feature_names
    )

    # Print top decision rules (interpretability advantage over RF and LR)
    print_tree_rules(model, feature_names, max_depth=3)

    # Plot feature importance
    plot_feature_importance(importance_df)

    # Cross-validation
    cv_scores = cross_validate_model(X, y, cv_folds=5)

    # Fairness evaluation
    race_test = df.loc[idx_test, 'race'].values
    fairness_results = evaluate_fairness(y_test, y_test_pred, y_test_proba, race_test)
    print()

    return model, feature_names, importance_df


if __name__ == "__main__":
    model, feature_names, importance_df = main()
