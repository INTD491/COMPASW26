"""
Random Forest Model for COMPAS Two-Year Recidivism Prediction

Key Literature & Citations:
---------------------------
1. Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). 
   "Machine Bias." ProPublica.
   
2. Tollenaar, N., & Van der Heijden, P. G. (2013). 
   "Which method predicts recidivism best?: a comparison of statistical, 
   machine learning and data mining predictive models." 
   Journal of the Royal Statistical Society, 176(2), 565-584.
   - Compared logistic regression, decision trees, random forests, and neural networks
   - Random forests showed best predictive performance
   
3. Brennan, T., Dieterich, W., & Ehret, B. (2009). 
   "Evaluating the predictive validity of the COMPAS risk and needs assessment system." 
   Criminal Justice and Behavior, 36(1), 21-40.
   
4. Liu, Y. Y., & Ridgeway, G. (2014). 
   "Tools and Techniques for Statistical Modeling in Criminal Justice." 
   RAND Corporation.
   - Recommends ensemble methods for criminal justice prediction
   
5. Duwe, G., & Rocque, M. (2017). 
   "Effects of automating recidivism risk assessment on reliability, predictive 
   validity, and return on investment (ROI)." 
   Criminology & Public Policy, 16(1), 235-269.

Random Forest Advantages:
-------------------------
- Handles non-linear relationships between features
- Captures feature interactions automatically
- More robust to outliers than logistic regression
- Provides feature importance rankings
- Less prone to overfitting with proper tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath):
    """
    Load and preprocess COMPAS dataset following ProPublica's filtering criteria.
    """
    df = pd.read_csv(filepath)
    
    # Apply ProPublica filters
    df = df[df['days_b_screening_arrest'] <= 30]
    df = df[df['days_b_screening_arrest'] >= -30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']
    
    return df


def prepare_features(df, include_race=False):
    """
    Prepare features for random forest model.
    Random forests don't require feature scaling but can benefit from it.
    """
    # Target variable
    y = df['two_year_recid'].values
    
    # Initialize features dictionary
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
    
    # Age categories (Random Forest can learn these patterns, but explicit encoding helps)
    age_cat_dummies = pd.get_dummies(df['age_cat'], prefix='age_cat')
    for col in age_cat_dummies.columns:
        features[col] = age_cat_dummies[col].values
    
    # Optional: Race (for fairness-aware models)
    if include_race:
        race_dummies = pd.get_dummies(df['race'], prefix='race')
        for col in race_dummies.columns:
            features[col] = race_dummies[col].values
    
    # Convert to DataFrame
    X = pd.DataFrame(features)
    feature_names = list(X.columns)
    
    return X.values, y, feature_names


def train_random_forest(X_train, y_train, tune_hyperparameters=False):
    """
    Train random forest model with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        tune_hyperparameters: Whether to perform grid search (slower but better)
    
    Returns:
        Trained random forest model
    """
    if tune_hyperparameters:
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, 
            param_grid, 
            cv=3, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    else:
        # Default parameters (reasonable baseline)
        rf = RandomForestClassifier(
            n_estimators=200,          # Number of trees
            max_depth=20,              # Maximum depth of trees
            min_samples_split=10,      # Minimum samples to split a node
            min_samples_leaf=4,        # Minimum samples at leaf node
            max_features='sqrt',       # Features to consider for splits
            class_weight='balanced',   # Handle class imbalance
            random_state=42,
            n_jobs=-1,                 # Use all CPU cores
            verbose=0
        )
        
        rf.fit(X_train, y_train)
        
        return rf


def evaluate_model(model, X_train, y_train, X_test, y_test, feature_names):
    """
    Comprehensive evaluation of the random forest model.
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
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Features:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:6.3f}")
    
    return y_test_pred, y_test_proba, importance_df


def plot_feature_importance(importance_df, save_path=None):
    """
    Plot feature importance from random forest.
    """
    plt.figure(figsize=(10, 6))
    
    top_features = importance_df.head(15)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance (Top 15)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    
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
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
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
    from sklearn.model_selection import cross_val_score
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores


def compare_to_baseline():
    """
    Print comparison to baseline logistic regression.
    """
    pass


def main():
    """
    Main execution function.
    """
    print("\n=== Random Forest Model ===")
    
    # Load data
    filepath = '../COMPASW26/datasets/compas-analysis/compas-scores-two-years.csv'
    df = load_and_preprocess_data(filepath)
    print(f"Samples: {len(df)} | Recidivism rate: {df['two_year_recid'].mean():.1%}")
    
    # Prepare features (race-blind model)
    X, y, feature_names = prepare_features(df, include_race=False)
    
    # Split data with indices to track demographics
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model (set tune_hyperparameters=True for better performance but slower)
    model = train_random_forest(X_train, y_train, tune_hyperparameters=False)
    
    # Evaluate model
    y_test_pred, y_test_proba, importance_df = evaluate_model(
        model, X_train, y_train, X_test, y_test, feature_names
    )
    
    # Plot feature importance
    plot_feature_importance(importance_df)
    
    # Cross-validation
    cv_scores = cross_validate_model(X, y, cv_folds=5)
    
    # Fairness evaluation
    race_test = df.loc[idx_test, 'race'].values
    fairness_metrics = evaluate_fairness(y_test, y_test_pred, y_test_proba, race_test)
    print()
    
    return model, feature_names, importance_df


if __name__ == "__main__":
    model, feature_names, importance_df = main()
