import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionGD:
    """
    Binary logistic regression trained with batch gradient descent.
    - L2 regularization via reg_lambda
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

        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.history_: list[tuple[int, float, float]] = []

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = sigmoid(X @ self.w + self.b)
        eps = 1e-12
        ce = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        reg = 0.5 * self.reg_lambda * float(np.sum(self.w ** 2))
        return float(ce + reg)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        n, d = X.shape
        self.w = self.rng.normal(0, 0.01, size=d)
        self.b = 0.0
        self.history_ = []

        best_val = float("inf")
        best_params = (self.w.copy(), self.b)
        no_improve = 0

        for it in range(1, self.max_iter + 1):
            z = X @ self.w + self.b
            p = sigmoid(z)

            grad_w = (X.T @ (p - y)) / n + self.reg_lambda * self.w
            grad_b = float(np.mean(p - y))

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            train_loss = self._loss(X, y)
            val_loss = self._loss(X_val, y_val)
            self.history_.append((it, train_loss, val_loss))

            if val_loss + self.tol < best_val:
                best_val = val_loss
                best_params = (self.w.copy(), self.b)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        self.w, self.b = best_params

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = sigmoid(X @ self.w + self.b)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)
