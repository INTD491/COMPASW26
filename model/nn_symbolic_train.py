import os
import importlib.util
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# PART 1: DATA LOADING AND PREPROCESSING     
def load_and_preprocess_compas():
    """Load and preprocess COMPAS dataset"""
    df = pd.read_csv('datasets/compas-analysis/compas-scores-two-years.csv')

    features = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'c_charge_degree', 'race', 'sex']
    target = 'two_year_recid'
    df = df[features + [target]].copy()

    df['c_charge_degree'] = df['c_charge_degree'].map({'F': 1, 'M': 0})
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['race_binary'] = (df['race'] == 'African-American').astype(int)
    df = df.drop('race', axis=1).dropna()

    X = df.drop(target, axis=1)
    y = df[target]

    # split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # fit scaler only on train
    scaler = StandardScaler()
    numerical_cols = ['age', 'priors_count', 'juv_fel_count', 'juv_misd_count']
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, y_train, y_test, scaler

# PART 2: SIMPLIFIED NEURO-SYMBOLIC MODEL (NO LTN - DIRECT IMPLEMENTATION)

class SimplifiedNeurosymbolicRecidivism(torch.nn.Module):
    """Simplified neuro-symbolic model without complex LTN dependencies"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
        # expanded rule banks
        self.high_rule_weights = torch.nn.Parameter(torch.ones(10) * 0.5)
        self.low_rule_weights = torch.nn.Parameter(torch.ones(8) * 0.5)
        self.alpha_logit = torch.nn.Parameter(torch.tensor(0.8))  # start with more rule influence

    # tuned thresholds from your quartiles
    def is_young(self, x):
        age = x[:, 0]
        return torch.sigmoid(-(age - (-0.8)) / 0.4)

    def is_older(self, x):
        age = x[:, 0]
        return torch.sigmoid((age - 0.7) / 0.4)

    def has_many_priors(self, x):
        priors = x[:, 1]
        return torch.sigmoid((priors - 0.3) / 0.35)

    def very_many_priors(self, x):
        priors = x[:, 1]
        return torch.sigmoid((priors - 0.9) / 0.35)

    def low_priors(self, x):
        priors = x[:, 1]
        return torch.sigmoid(-(priors - (-0.3)) / 0.35)

    def no_priors(self, x):
        priors = x[:, 1]
        return torch.sigmoid(-(priors - (-0.75)) / 0.3)

    def has_juvenile_record(self, x):
        juv_total = x[:, 2] + x[:, 3]
        return torch.sigmoid((juv_total - (-0.1)) / 0.5)

    def no_juvenile_record(self, x):
        juv_total = x[:, 2] + x[:, 3]
        return torch.sigmoid(-(juv_total - (-0.6)) / 0.5)

    def is_felony(self, x):
        return torch.sigmoid((x[:, 4] - 0.5) / 0.15)

    def is_misdemeanor(self, x):
        return 1.0 - self.is_felony(x)

    def is_male(self, x):
        return x[:, 5]

    def is_female(self, x):
        return 1.0 - self.is_male(x)

    def apply_symbolic_rules(self, x):
        high_rules = torch.zeros(x.shape[0], 10, device=x.device)
        high_rules[:, 0] = self.is_young(x) * self.has_many_priors(x)
        high_rules[:, 1] = self.is_felony(x) * self.has_many_priors(x)
        high_rules[:, 2] = self.is_male(x) * self.has_many_priors(x)
        high_rules[:, 3] = self.is_young(x) * self.is_felony(x) * self.has_many_priors(x)
        high_rules[:, 4] = self.has_juvenile_record(x) * self.has_many_priors(x)
        high_rules[:, 5] = self.very_many_priors(x)
        high_rules[:, 6] = self.is_young(x) * self.is_felony(x)
        high_rules[:, 7] = self.is_male(x) * self.is_felony(x) * self.has_many_priors(x)
        high_rules[:, 8] = self.has_juvenile_record(x)
        high_rules[:, 9] = self.is_young(x) * self.has_juvenile_record(x)

        low_rules = torch.zeros(x.shape[0], 8, device=x.device)
        low_rules[:, 0] = self.is_older(x) * self.no_juvenile_record(x)
        low_rules[:, 1] = self.is_older(x) * self.is_misdemeanor(x)
        low_rules[:, 2] = self.low_priors(x) * self.no_juvenile_record(x)
        low_rules[:, 3] = self.no_priors(x) * self.no_juvenile_record(x)
        low_rules[:, 4] = self.is_female(x) * self.no_priors(x)
        low_rules[:, 5] = self.is_older(x) * self.low_priors(x)
        low_rules[:, 6] = self.is_misdemeanor(x) * self.no_juvenile_record(x)
        low_rules[:, 7] = self.is_older(x) * self.no_priors(x)

        return high_rules, low_rules
    
    def forward(self, x):
        neural_pred = self.neural_net(x).squeeze()
        high_rules, low_rules = self.apply_symbolic_rules(x)
        high_score = torch.matmul(high_rules, torch.nn.functional.softmax(self.high_rule_weights, dim=0))
        low_score = torch.matmul(low_rules, torch.nn.functional.softmax(self.low_rule_weights, dim=0))
        rule_prob = torch.sigmoid(2.2 * (high_score - low_score))

        alpha = torch.sigmoid(self.alpha_logit)  # learned [0,1]
        return alpha * neural_pred + (1 - alpha) * rule_prob
    
    def explain(self, x):
        with torch.no_grad():
            neural_pred = self.neural_net(x).squeeze()
            high_rules, low_rules = self.apply_symbolic_rules(x)
            final_pred = self.forward(x)

            all_rules = torch.cat([high_rules, low_rules], dim=1)

            return {
                "neural_score": neural_pred.item() if x.shape[0] == 1 else neural_pred.cpu().numpy(),
                "rule_scores": all_rules.cpu().numpy(),  # keeps your existing print working
                "high_rule_scores": high_rules.cpu().numpy(),
                "low_rule_scores": low_rules.cpu().numpy(),
                "final_prediction": final_pred.item() if x.shape[0] == 1 else final_pred.cpu().numpy()
            }

def fuzzy_implies(a, b):
    # Product logic implication approximation
    return torch.clamp(1.0 - a + a * b, 0.0, 1.0)

def logic_satisfaction_loss(model, x):
    """
    True neural-symbolic step:
    optimize logical formula satisfaction, not only rule feature blending.
    """
    p = model(x)  # predicted recidivism probability

    # F1: young & many_priors -> recid
    ant1 = model.is_young(x) * model.has_many_priors(x)
    f1 = fuzzy_implies(ant1, p)

    # F2: felony & many_priors -> recid
    ant2 = model.is_felony(x) * model.has_many_priors(x)
    f2 = fuzzy_implies(ant2, p)

    # F3: older & no_juvenile & low_priors -> not recid
    ant3 = model.is_older(x) * model.no_juvenile_record(x) * model.low_priors(x)
    f3 = fuzzy_implies(ant3, 1.0 - p)

    # F4: no_priors & no_juvenile -> not recid
    ant4 = model.no_priors(x) * model.no_juvenile_record(x)
    f4 = fuzzy_implies(ant4, 1.0 - p)

    sat = torch.stack([f1, f2, f3, f4], dim=1).mean()  # maximize
    return 1.0 - sat, sat

# PART 3: TRAINING WITH FAIRNESS CONSTRAINTS
def fairness_loss(predictions, race_binary, lambda_fairness=0.1):
    """Compute fairness loss to minimize disparate impact"""
    
    # Separate by race
    protected_group = race_binary == 1  # African-American
    privileged_group = race_binary == 0  # Other
    
    if protected_group.sum() == 0 or privileged_group.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    # Calculate positive prediction rates
    protected_rate = predictions[protected_group].mean()
    privileged_rate = predictions[privileged_group].mean()
    
    # Minimize difference (demographic parity)
    fairness_penalty = torch.abs(protected_rate - privileged_rate)
    
    return lambda_fairness * fairness_penalty

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, class_weights=None):
    """Train the neuro-symbolic model"""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = torch.nn.BCELoss(reduction='none')

    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_val_tensor = torch.FloatTensor(y_val.values)

    if class_weights is None:
        pos_w = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        neg_w = 1.0
    else:
        neg_w, pos_w = class_weights

    sample_weights = y_train_tensor * pos_w + (1 - y_train_tensor) * neg_w
    race_train = torch.FloatTensor(X_train['race_binary'].values)

    best_val_auc = -1.0
    patience, patience_counter = 25, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(X_train_tensor)
        data_loss = (criterion(pred, y_train_tensor) * sample_weights).mean()

        logic_loss, sat = logic_satisfaction_loss(model, X_train_tensor)

        lam_logic = min(0.35, 0.35 * (epoch + 1) / 100.0)
        lam_fair = min(0.10, 0.10 * (epoch + 1) / 100.0)
        fair_loss = fairness_loss(pred, race_train, lambda_fairness=lam_fair)

        total_loss = data_loss + lam_logic * logic_loss + fair_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                vp = model(X_val_tensor)
                vb = (vp > 0.5).float().cpu().numpy()
                va = roc_auc_score(y_val_tensor.cpu().numpy(), vp.cpu().numpy())
                vacc = accuracy_score(y_val_tensor.cpu().numpy(), vb)

            scheduler.step(va)
            print(f"Epoch {epoch+1}/{epochs} | AUC={va:.4f} Acc={vacc:.4f} "
                  f"Loss={total_loss.item():.4f} Data={data_loss.item():.4f} "
                  f"LogicSat={sat.item():.4f} LR={optimizer.param_groups[0]['lr']:.2e}")

            if va > best_val_auc:
                best_val_auc = va
                patience_counter = 0
                torch.save(model.state_dict(), "best_model_auc.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    model.load_state_dict(torch.load("best_model_auc.pth", map_location="cpu"))
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    return model

# PART 4: EVALUATION AND EXPLAINABILITY

def evaluate_fairness(model, X_test, y_test):
    """Evaluate model fairness metrics"""
    
    X_test_tensor = torch.FloatTensor(X_test.values)
    
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions_binary = (predictions > 0.5).float().numpy()
    
    race = X_test['race_binary'].values
    
    # Calculate metrics for each group
    protected_mask = race == 1
    privileged_mask = race == 0
    
    protected_positive_rate = predictions_binary[protected_mask].mean()
    privileged_positive_rate = predictions_binary[privileged_mask].mean()
    
    disparate_impact = protected_positive_rate / (privileged_positive_rate + 1e-10)
    
    print("\n" + "="*50)
    print("FAIRNESS EVALUATION")
    print("="*50)
    print(f"Protected Group (African-American) Positive Rate: {protected_positive_rate:.3f}")
    print(f"Privileged Group (Other) Positive Rate: {privileged_positive_rate:.3f}")
    print(f"Disparate Impact Ratio: {disparate_impact:.3f}")
    print("  (Ideal: 0.8 - 1.2, closer to 1.0 is more fair)")
    
    # Equal opportunity (TPR parity)
    protected_tpr = predictions_binary[protected_mask & (y_test.values == 1)].mean() if (protected_mask & (y_test.values == 1)).sum() > 0 else 0
    privileged_tpr = predictions_binary[privileged_mask & (y_test.values == 1)].mean() if (privileged_mask & (y_test.values == 1)).sum() > 0 else 0
    
    print(f"\nTrue Positive Rate (Protected): {protected_tpr:.3f}")
    print(f"True Positive Rate (Privileged): {privileged_tpr:.3f}")
    print(f"TPR Difference: {abs(protected_tpr - privileged_tpr):.3f}")
    
    return disparate_impact

def full_evaluation(model, X_test, y_test):
    """Comprehensive model evaluation"""
    
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions_binary = (predictions > 0.5).float().numpy().flatten()
    
    print("\n" + "="*50)
    print("PERFORMANCE EVALUATION")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions_binary, 
                                target_names=['No Recidivism', 'Recidivism']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions_binary))
    
    auc = roc_auc_score(y_test, predictions.numpy())
    print(f"\nAUC-ROC Score: {auc:.4f}")
    
    # Fairness evaluation
    evaluate_fairness(model, X_test, y_test)
    
    # Example explanations
    print("\n" + "="*50)
    print("EXAMPLE EXPLANATIONS")
    print("="*50)
    for i in range(min(3, len(X_test))):
        sample = X_test.iloc[i:i+1]
        sample_tensor = torch.FloatTensor(sample.values)
        exp = model.explain(sample_tensor)
        
        print(f"\nSample {i+1}:")
        print(f"  Neural Score: {exp['neural_score']:.3f}")
        print(f"  Rule Scores: {exp['rule_scores'][0]}")
        print(f"  Final Prediction: {'High Risk' if exp['final_prediction'] > 0.5 else 'Low Risk'} ({exp['final_prediction']:.3f})")

# PART 5: PATTERN MINING

def mine_dataset_patterns(X, y):
    """Print simple correlations and interaction rates to design symbolic rules."""
    df = X.copy()
    df["target"] = y.values

    print("\n" + "="*50)
    print("PATTERN MINING (TRAIN SPLIT)")
    print("="*50)

    # Linear correlations
    corr = df.corr(numeric_only=True)["target"].sort_values(ascending=False)
    print("\n[Correlation with target]")
    print(corr)

    # Binned numeric rates
    for col in ["age", "priors_count", "juv_fel_count", "juv_misd_count"]:
        try:
            b = pd.qcut(df[col], q=4, duplicates="drop")
            rates = df.groupby(b)["target"].mean()
            print(f"\n[{col}] recidivism rate by quartile:")
            print(rates)
        except Exception:
            pass

    # Binary features rates
    for col in ["sex", "c_charge_degree", "race_binary"]:
        if col in df.columns:
            rates = df.groupby(col)["target"].mean()
            print(f"\n[{col}] recidivism rate by value:")
            print(rates)

    # Useful interactions
    checks = {
        "young & many_priors": ((df["age"] < df["age"].quantile(0.25)) & (df["priors_count"] > df["priors_count"].median())),
        "felony & many_priors": ((df["c_charge_degree"] == 1) & (df["priors_count"] > df["priors_count"].median())),
        "male & many_priors": ((df["sex"] == 1) & (df["priors_count"] > df["priors_count"].median())),
        "older & no_juvenile": ((df["age"] > df["age"].quantile(0.75)) & ((df["juv_fel_count"] + df["juv_misd_count"]) <= 0)),
    }
    print("\n[Interaction condition rates]")
    for name, mask in checks.items():
        if mask.sum() > 0:
            print(f"{name}: n={int(mask.sum())}, recid_rate={df.loc[mask, 'target'].mean():.4f}")

# MAIN EXECUTION

def main():
    print("Loading and preprocessing COMPAS dataset...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_compas()

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {list(X_train.columns)}")

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    mine_dataset_patterns(X_train_split, y_train_split)

    print("\nInitializing Neuro-Symbolic model...")
    model = SimplifiedNeurosymbolicRecidivism(input_dim=X_train.shape[1])

    # Standalone training mode (no cloud_host import)
    print("[INFO] Standalone training mode: using imbalance-derived class weights.")
    # Optional: uncomment to warm-start from a saved checkpoint
    # init_ckpt = "recidivism_neurosymbolic_model.pth"
    # if os.path.exists(init_ckpt):
    #     model.load_state_dict(torch.load(init_ckpt, map_location="cpu"), strict=False)
    #     print(f"[INFO] Loaded initial weights from {init_ckpt}")

    print("\nTraining model with symbolic rules and fairness constraints...")
    model = train_model(
        model,
        X_train_split,
        y_train_split,
        X_val,
        y_val,
        epochs=800,
        lr=0.001,
        class_weights=None
    )

    print("\nEvaluating on test set...")
    full_evaluation(model, X_test, y_test)

    torch.save(model.state_dict(), 'recidivism_neurosymbolic_model.pth')
    print("\nModel saved to 'recidivism_neurosymbolic_model.pth'")

if __name__ == "__main__":
    main()