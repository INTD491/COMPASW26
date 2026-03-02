"""
Baseline Logistic Regression Model for COMPAS Two-Year Recidivism Prediction

Key Literature & Citations:
---------------------------
1. Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). 
   "Machine Bias: There's software used across the country to predict future criminals. 
   And it's biased against blacks." ProPublica.
   
2. Dressel, J., & Farid, H. (2018). 
   "The accuracy, fairness, and limits of predicting recidivism." 
   Science advances, 4(1), eaao5580.
   - Showed that simple linear models can match COMPAS accuracy
   - Used features: age, sex, prior convictions, charge degree
   
3. Rudin, C., Wang, C., & Coker, B. (2020). 
   "The age of secrecy and unfairness in recidivism prediction." 
   Harvard Data Science Review, 2(1).
   - Advocates for interpretable models over black-box approaches
   
4. Corbett-Davies, S., Pierson, E., Feller, A., Goel, S., & Huq, A. (2017). 
   "Algorithmic decision making and the cost of fairness." 
   Proceedings of the 23rd ACM SIGKDD, 797-806.
   
5. Dieterich, W., Mendoza, C., & Brennan, T. (2016). 
   "COMPAS risk scales: Demonstrating accuracy equity and predictive parity."
   Northpointe Inc.

Standard Features in Literature:
--------------------------------
- age / age_cat: Age at screening (strong predictor, younger = higher risk)
- sex: Gender (male typically higher risk)
- race: Used for fairness analysis (not always included as predictor)
- priors_count: Number of prior convictions (strongest predictor)
- juv_fel_count: Juvenile felony count
- juv_misd_count: Juvenile misdemeanor count  
- juv_other_count: Juvenile other offense count
- c_charge_degree: Current charge degree (Felony vs Misdemeanor)
- days_b_screening_arrest: Days between screening and arrest (data quality)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath):
    """
    Load and preprocess COMPAS dataset following ProPublica's filtering criteria.
    
    ProPublica filtering (2016):
    - Remove cases where days_b_screening_arrest is over 30 or under -30
    - Remove cases where is_recid is -1
    - Remove cases where c_charge_degree is 'O' (ordinary traffic offense)
    - Keep only Compas screening date between 2013-2014
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
    Prepare features following Dressel & Farid (2018) and common literature.
    
    Args:
        df: DataFrame with COMPAS data
        include_race: Whether to include race as a feature (for fairness analysis)
    
    Returns:
        X: Feature matrix
        y: Target variable (two_year_recid)
        feature_names: List of feature names
    """
    # Target variable
    y = df['two_year_recid'].values
    
    # Initialize features dictionary
    features = {}
    
    # 1. Age (continuous) - normalized
    features['age'] = df['age'].values
    
    # 2. Sex (binary: Male=1, Female=0)
    features['sex_male'] = (df['sex'] == 'Male').astype(int).values
    
    # 3. Prior convictions (strongest predictor)
    features['priors_count'] = df['priors_count'].values
    
    # 4. Juvenile history
    features['juv_fel_count'] = df['juv_fel_count'].values
    features['juv_misd_count'] = df['juv_misd_count'].values
    features['juv_other_count'] = df['juv_other_count'].values
    
    # 5. Current charge degree (Felony=1, Misdemeanor=0)
    features['charge_degree_felony'] = (df['c_charge_degree'] == 'F').astype(int).values
    
    # 6. Age categories (often used as alternative to continuous age)
    age_cat_dummies = pd.get_dummies(df['age_cat'], prefix='age_cat')
    for col in age_cat_dummies.columns:
        features[col] = age_cat_dummies[col].values
    
    # 7. Optional: Race (for fairness-aware models)
    if include_race:
        race_dummies = pd.get_dummies(df['race'], prefix='race')
        for col in race_dummies.columns:
            features[col] = race_dummies[col].values
    
    # Convert to DataFrame for easy handling
    X = pd.DataFrame(features)
    feature_names = list(X.columns)
    
    return X.values, y, feature_names


def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train logistic regression baseline model.
    
    Following Dressel & Farid (2018): Simple linear classifier with L2 regularization.
    """
    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression with L2 regularization
    # C=1.0 is default, balanced class weights to handle imbalance
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test, feature_names):
    """
    Comprehensive evaluation of the model.
    """
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f} | ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Feature importance (coefficients)
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0]
    })
    coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.sort_values('abs_coef', ascending=False)
    
    print("\nTop 5 Features:")
    for idx, row in coef_df.head(5).iterrows():
        print(f"  {row['feature']:25s}: {row['coefficient']:6.3f}")
    
    return y_pred, y_pred_proba


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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    # Stratified K-Fold to maintain class distribution
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
    
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores


def main():
    """
    Main execution function.
    """
    print("\n=== Baseline Logistic Regression ===")
    
    # Load data
    filepath = 'datasets/compas-analysis/compas-scores-two-years.csv'
    df = load_and_preprocess_data(filepath)
    print(f"Samples: {len(df)} | Recidivism rate: {df['two_year_recid'].mean():.1%}")
    
    # Prepare features (without race - race-blind model)
    X, y, feature_names = prepare_features(df, include_race=False)
    
    # Split data with indices to track demographics
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model, scaler = train_baseline_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    y_pred, y_pred_proba = evaluate_model(model, scaler, X_test, y_test, feature_names)
    
    # Cross-validation
    cv_scores = cross_validate_model(X, y, cv_folds=5)
    
    # Fairness evaluation
    race_test = df.loc[idx_test, 'race'].values
    fairness_metrics = evaluate_fairness(y_test, y_pred, y_pred_proba, race_test)
    print()
    
    return model, scaler, feature_names


if __name__ == "__main__":
    model, scaler, feature_names = main()
