"""
ImAUC-PSVM: Imbalanced AUC-Proximal Support Vector Machine (2026)

Optimizes AUC directly via a pairwise ranking formulation solved with
proximal SVM efficiency (closed-form linear system instead of QP).

Key idea:
    Standard SVMs optimise accuracy by finding a separating hyperplane.
    ImAUC-PSVM instead maximises the probability that a randomly chosen
    positive sample is scored higher than a randomly chosen negative sample
    — i.e., it directly maximises AUC.

    Given positive samples X+ (n+ x d) and negative samples X- (n- x d),
    form the *virtual* pairwise difference matrix Z whose rows are
    (x_i+ - x_j-) for every (i,j) pair.  The objective is:

        min_{w,b}  (1/2) ||Z w + e b - e||^2  +  (C/2) ||w||^2

    which has the closed-form normal-equation solution:

        [ Z^T Z + C I    Z^T e ] [ w ]   [ Z^T e ]
        [ e^T Z          m     ] [ b ] = [   m   ]

    where m = n+ * n-.

    The key computational trick is that Z^T Z, Z^T e, etc. can be
    computed from class-level statistics (O(n d^2)) without ever
    materialising the full m x d pairwise matrix (O(n+*n- * d)).

Requires feature scaling (handled by train.py).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ImAUCPSVM(BaseEstimator, ClassifierMixin):
    """
    Imbalanced AUC-Proximal Support Vector Machine.

    Parameters
    ----------
    C : float, default=1.0
        Regularisation strength (higher → more regularisation, simpler model).
    """

    _estimator_type = "classifier"

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()

        X_pos = X[y == 1]
        X_neg = X[y == 0]
        n_pos, d = X_pos.shape
        n_neg = X_neg.shape[0]
        m = n_pos * n_neg  # number of virtual pairs

        # ------------------------------------------------------------------
        # Efficient computation of Z^T Z and Z^T e without forming Z (m x d)
        # ------------------------------------------------------------------
        # Z^T Z = n- X+^T X+ + n+ X-^T X- - s+ s-^T - s- s+^T
        S_pos = X_pos.T @ X_pos                       # d x d
        S_neg = X_neg.T @ X_neg                       # d x d
        s_pos = X_pos.sum(axis=0).reshape(-1, 1)       # d x 1
        s_neg = X_neg.sum(axis=0).reshape(-1, 1)       # d x 1

        ZTZ = (n_neg * S_pos + n_pos * S_neg
               - s_pos @ s_neg.T - s_neg @ s_pos.T)

        # Z^T e = n- s+ - n+ s-
        ZTe = (n_neg * s_pos - n_pos * s_neg).ravel()  # d

        # ------------------------------------------------------------------
        # In pairwise differences the bias cancels:
        #   f(x+) - f(x-) = (w'x+ + b) - (w'x- + b) = w'(x+ - x-)
        # So the objective reduces to a bias-free system:
        #
        #   min_{w}  (1/2m) ||Z w - e||^2  +  (C/2) ||w||^2
        #
        # Normal equations (normalised by m for conditioning):
        #
        #   [(1/m) Z^T Z  +  C I] w  =  (1/m) Z^T e
        # ------------------------------------------------------------------
        A = ZTZ / m + self.C * np.eye(d)
        rhs = ZTe / m

        self.w_ = np.linalg.solve(A, rhs)

        # Bias is selected post-hoc: find the threshold on training scores
        # that best separates the classes (maximises Youden's J = TPR - FPR).
        train_scores = X @ self.w_
        self.b_ = self._find_bias(train_scores, y)

        # ------------------------------------------------------------------
        # Calibrate sigmoid mapping  score → probability
        # Fit Platt scaling:  p = 1/(1+exp(-(a*s + b)))
        # ------------------------------------------------------------------
        train_scores = self.decision_function(X)
        self._fit_platt(train_scores, y)

        self.classes_ = np.array([0, 1])
        return self

    def _find_bias(self, scores, y):
        """Find bias that maximises Youden's J statistic (TPR - FPR)."""
        # Try thresholds at each unique score midpoint
        pos_scores = scores[y == 1]
        neg_scores = scores[y == 0]
        thresholds = np.percentile(scores, np.arange(5, 96, 1))

        best_j, best_t = -1, np.median(scores)
        for t in thresholds:
            tpr = (pos_scores >= t).mean()
            fpr = (neg_scores >= t).mean()
            j = tpr - fpr
            if j > best_j:
                best_j, best_t = j, t
        # bias so that decision_function(x) > 0 ↔ w'x > threshold
        return -best_t

    # --- Platt scaling (sigmoid calibration) ----------------------------

    def _fit_platt(self, scores, y):
        """Fit sigmoid parameters via maximum-likelihood (Newton's method)."""
        from scipy.optimize import minimize

        # Target probabilities with Bayesian correction
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        t_pos = (n_pos + 1) / (n_pos + 2)
        t_neg = 1 / (n_neg + 2)
        target = np.where(y == 1, t_pos, t_neg)

        def loss(params):
            a, b = params
            p = 1 / (1 + np.exp(-(a * scores + b)))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p))

        result = minimize(loss, x0=[1.0, 0.0], method="L-BFGS-B")
        self.platt_a_, self.platt_b_ = result.x

    def _sigmoid(self, scores):
        z = self.platt_a_ * scores + self.platt_b_
        return 1 / (1 + np.exp(-z))

    # --- Prediction interface (scikit-learn compatible) -----------------

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w_ + self.b_

    def predict_proba(self, X):
        scores = self.decision_function(X)
        p1 = self._sigmoid(scores)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


# -----------------------------------------------------------------------
# Interface expected by train.py
# -----------------------------------------------------------------------
def build_model():
    return ImAUCPSVM(C=1.0)


NEEDS_SCALING = True
