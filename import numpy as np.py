import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss
)

# ----------------------------
# From-scratch Logistic Regression
# ----------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))

def logit(p: np.ndarray) -> np.ndarray:
    # stable logit
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
class PlattCalibrator:
    """
    Platt scaling: fit sigmoid(a*s + c) on validation data,
    where s is the base model logit score.
    """
    def __init__(self, lr: float = 0.05, max_iter: int = 10000, tol: float = 1e-8, seed: int = 42):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.a = None
        self.c = None

    def fit(self, s_val: np.ndarray, y_val: np.ndarray):
        n = s_val.shape[0]
        # initialize close to identity
        self.a = 1.0
        self.c = 0.0
        best_loss = float("inf")
        no_improve = 0

        for _ in range(self.max_iter):
            p = sigmoid(self.a * s_val + self.c)
            eps = 1e-12
            loss = -np.mean(y_val * np.log(p + eps) + (1 - y_val) * np.log(1 - p + eps))

            # gradients
            # dL/da = mean((p - y) * s)
            # dL/dc = mean(p - y)
            grad_a = np.mean((p - y_val) * s_val)
            grad_c = np.mean(p - y_val)

            self.a -= self.lr * grad_a
            self.c -= self.lr * grad_c

            if loss + self.tol < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 200:
                    break

    def predict_proba(self, s: np.ndarray) -> np.ndarray:
        p1 = sigmoid(self.a * s + self.c)
        return np.vstack([1 - p1, p1]).T
    
def calibration_report(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """
    Returns:
      ece: expected calibration error (weighted by bin frequency)
      table: list of (bin_lo, bin_hi, count, avg_conf, frac_pos)
    """
    p = np.asarray(p).astype(float)
    y = np.asarray(y).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    table = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # include right edge only in last bin
        if i < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)

        cnt = int(np.sum(mask))
        if cnt == 0:
            table.append((lo, hi, 0, np.nan, np.nan))
            continue

        avg_conf = float(np.mean(p[mask]))
        frac_pos = float(np.mean(y[mask]))  # empirical P(y=1 | bin)
        ece += (cnt / len(p)) * abs(avg_conf - frac_pos)
        table.append((lo, hi, cnt, avg_conf, frac_pos))

    return float(ece), table

    

class LogisticRegressionGD:
    """
    Binary logistic regression trained with batch gradient descent.

    Supports:
    - L2 regularization (ridge) via reg_lambda
    - early stopping on validation loss
    """
    def __init__(
        self,
        lr: float = 0.1,
        max_iter: int = 5000,
        reg_lambda: float = 0.0,
        tol: float = 1e-6,
        patience: int = 30,
        seed: int = 42,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        self.w = None
        self.b = 0.0
        self.history_ = []

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = sigmoid(X @ self.w + self.b)
        # logistic loss (negative log-likelihood)
        eps = 1e-12
        ce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        # L2 regularization (exclude bias)
        reg = 0.5 * self.reg_lambda * np.sum(self.w ** 2)
        return ce + reg

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        n, d = X.shape
        self.w = self.rng.normal(0, 0.01, size=d)
        self.b = 0.0

        best_val = float("inf")
        best_params = (self.w.copy(), self.b)
        no_improve = 0

        for it in range(1, self.max_iter + 1):
            # forward
            z = X @ self.w + self.b
            p = sigmoid(z)

            # gradients
            # d/dw: X^T (p - y)/n  + lambda*w
            grad_w = (X.T @ (p - y)) / n + self.reg_lambda * self.w
            grad_b = np.mean(p - y)

            # update
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            # monitor
            train_loss = self._loss(X, y)
            val_loss = self._loss(X_val, y_val)
            self.history_.append((it, train_loss, val_loss))

            # early stopping on validation loss
            if val_loss + self.tol < best_val:
                best_val = val_loss
                best_params = (self.w.copy(), self.b)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        # restore best
        self.w, self.b = best_params

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = sigmoid(X @ self.w + self.b)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

# ----------------------------
# Experiment: Breast Cancer dataset
# ----------------------------
def main():
    data = load_breast_cancer()
    X = data["data"].astype(np.float64)
    y = data["target"].astype(int)  # 0/1

    # split: train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # standardize (important for GD)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # train from scratch model
    model = LogisticRegressionGD(
        lr=0.2,
        max_iter=20000,
        reg_lambda=0.01,
        tol=1e-7,
        patience=50,
        seed=42,
    )
    model.fit(X_train_s, y_train, X_val_s, y_val)

    # --- Calibration on validation set (Platt scaling) ---
    p_val = model.predict_proba(X_val_s)[:, 1]
    s_val = logit(p_val)  # base score in logit space

    cal = PlattCalibrator(lr=0.05, max_iter=20000, tol=1e-10, seed=42)
    cal.fit(s_val, y_val)


    # evaluate
    # Uncalibrated
    p_test = model.predict_proba(X_test_s)[:, 1]
    y_pred = (p_test >= 0.5).astype(int)

    # Calibrated
    s_test = logit(p_test)
    p_test_cal = cal.predict_proba(s_test)[:, 1]
    y_pred_cal = (p_test_cal >= 0.5).astype(int)

    ece_u, tab_u = calibration_report(p_test, y_test, n_bins=10)
    ece_c, tab_c = calibration_report(p_test_cal, y_test, n_bins=10)

    print("Calibration diagnostics (Test set)")
    print("----------------------------------")
    print(f"ECE uncalibrated: {ece_u:.4f}")
    print(f"ECE calibrated:   {ece_c:.4f}")
    print()

    def print_table(tab, title):
        print(title)
        print("bin_range      count  avg_conf  frac_pos")
        for lo, hi, cnt, avg_conf, frac_pos in tab:
            if cnt == 0:
                print(f"[{lo:.1f},{hi:.1f})".ljust(13) + f"{cnt:>7d}  {'-':>8}  {'-':>8}")
            else:
                print(f"[{lo:.1f},{hi:.1f})".ljust(13) + f"{cnt:>7d}  {avg_conf:>8.3f}  {frac_pos:>8.3f}")
        print()

    print_table(tab_u, "Uncalibrated reliability table")
    print_table(tab_c, "Calibrated reliability table")


   

    print("From-scratch Logistic Regression (Uncalibrated)")
    print("------------------------------------------------")
    print(f"Iterations: {len(model.history_)}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test ROC-AUC:  {roc_auc_score(y_test, p_test):.4f}")
    print(f"Test LogLoss:  {log_loss(y_test, p_test):.4f}")
    print(f"Test Brier:    {brier_score_loss(y_test, p_test):.4f}")
    print()

    print("After Platt Scaling (Calibrated on Validation)")
    print("----------------------------------------------")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_cal):.4f}")
    print(f"Test ROC-AUC:  {roc_auc_score(y_test, p_test_cal):.4f}")
    print(f"Test LogLoss:  {log_loss(y_test, p_test_cal):.4f}")
    print(f"Test Brier:    {brier_score_loss(y_test, p_test_cal):.4f}")
    print(f"Platt params:  a={cal.a:.4f}, c={cal.c:.4f}")
    print()


    # quick peek: last few losses
    tail = model.history_[-5:]
    print("Last 5 (iter, train_loss, val_loss):")
    for it, tr, va in tail:
        print(f"{it:>6d}  {tr:.6f}  {va:.6f}")

if __name__ == "__main__":
    main()
