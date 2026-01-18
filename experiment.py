from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .model import LogisticRegressionGD


# ----------------------------
# Helpers: numeric stability
# ----------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """
    Platt scaling: fit sigma(a*s + c) on validation set, where s is base logit score.
    Trained via simple batch GD on cross-entropy.
    """
    def __init__(self, lr: float = 0.05, max_iter: int = 20000, tol: float = 1e-10, patience: int = 200):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.a = 1.0
        self.c = 0.0

    def fit(self, s_val: np.ndarray, y_val: np.ndarray):
        y_val = y_val.astype(int)
        best = float("inf")
        no_improve = 0

        for _ in range(self.max_iter):
            p = _sigmoid(self.a * s_val + self.c)
            eps = 1e-12
            loss = -np.mean(y_val * np.log(p + eps) + (1 - y_val) * np.log(1 - p + eps))

            # grads
            grad_a = np.mean((p - y_val) * s_val)
            grad_c = np.mean(p - y_val)

            self.a -= self.lr * grad_a
            self.c -= self.lr * grad_c

            if loss + self.tol < best:
                best = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

    def predict_proba(self, s: np.ndarray) -> np.ndarray:
        p1 = _sigmoid(self.a * s + self.c)
        return np.vstack([1 - p1, p1]).T


@dataclass
class RunConfig:
    # from-scratch logistic regression hyperparams
    lr: float = 0.2
    max_iter: int = 20000
    reg_lambda: float = 0.01
    tol: float = 1e-7
    patience: int = 50
    seed: int = 42
    test_size_total: float = 0.30
    val_share_of_temp: float = 0.50  # val/test split within temp
    threshold: float = 0.50

    # controlled difficulty knobs (train-only)
    label_noise_rate: float = 0.0        # fraction of train labels flipped
    feature_noise_std: float = 0.0       # Gaussian noise added to train features (raw space)

    # benchmark
    run_benchmark: bool = True
    rf_n_estimators: int = 400
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 1

    # calibration
    use_platt: bool = False
    platt_lr: float = 0.05
    platt_max_iter: int = 20000
    platt_tol: float = 1e-10
    platt_patience: int = 200


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def _apply_label_noise(y: np.ndarray, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """
    Flip a fraction of labels in y (binary 0/1). Train-only.
    Returns (noisy_y, n_flipped).
    """
    if rate <= 0.0:
        return y.copy(), 0
    rate = float(np.clip(rate, 0.0, 0.5))  # >0.5 makes no sense for binary flips
    n = y.shape[0]
    k = int(round(rate * n))
    if k == 0:
        return y.copy(), 0
    idx = rng.choice(n, size=k, replace=False)
    y2 = y.copy()
    y2[idx] = 1 - y2[idx]
    return y2, k


def _apply_feature_noise(X: np.ndarray, std: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """
    Add Gaussian noise N(0, std^2) to features. Train-only.
    Returns (noisy_X, realized_std).
    """
    if std <= 0.0:
        return X.copy(), 0.0
    std = float(max(0.0, std))
    noise = rng.normal(0.0, std, size=X.shape)
    return (X + noise), std


def run_and_log(config: RunConfig, runs_dir: str = "runs") -> Path:
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)

    data = load_breast_cancer()
    X = data["data"].astype(np.float64)
    y = data["target"].astype(int)
    feature_names = list(data["feature_names"])

    # train / val / test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=config.test_size_total,
        random_state=config.seed,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=config.val_share_of_temp,
        random_state=config.seed,
        stratify=y_temp,
    )

    # Apply controlled difficulty knobs (TRAIN ONLY)
    X_train_noisy, realized_feature_noise = _apply_feature_noise(X_train, config.feature_noise_std, rng)
    y_train_noisy, n_flipped = _apply_label_noise(y_train, config.label_noise_rate, rng)

    # standardize (fit on train only) using noisy train features (this is deliberate)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_noisy)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ----------------------------
    # Benchmark: RandomForest
    # ----------------------------
    bench = None
    bench_metrics = None
    p_test_b = None
    if config.run_benchmark:
        bench = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            random_state=config.seed,
            n_jobs=-1,
        )
        # Use *raw* features; train on the same noisy train features for fairness when feature_noise_std > 0
        bench.fit(X_train_noisy, y_train_noisy)
        p_test_b = bench.predict_proba(X_test)[:, 1]
        y_pred_b = (p_test_b >= config.threshold).astype(int)

        bench_metrics = {
            "accuracy": accuracy_score(y_test, y_pred_b),
            "roc_auc": roc_auc_score(y_test, p_test_b),
            "log_loss": log_loss(y_test, p_test_b),
            "brier": brier_score_loss(y_test, p_test_b),
        }

    # ----------------------------
    # From-scratch Logistic Regression
    # ----------------------------
    model = LogisticRegressionGD(
        lr=config.lr,
        max_iter=config.max_iter,
        reg_lambda=config.reg_lambda,
        tol=config.tol,
        patience=config.patience,
        seed=config.seed,
    )
    model.fit(X_train_s, y_train_noisy, X_val_s, y_val)

    p_test = model.predict_proba(X_test_s)[:, 1]
    y_pred = (p_test >= config.threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, p_test),
        "log_loss": log_loss(y_test, p_test),
        "brier": brier_score_loss(y_test, p_test),
    }

    # ----------------------------
    # Optional: Platt scaling calibration (fit on validation)
    # ----------------------------
    cal = None
    cal_metrics = None
    p_test_cal = None
    platt_params = None
    if config.use_platt:
        p_val = model.predict_proba(X_val_s)[:, 1]
        s_val = _logit(p_val)
        cal = PlattCalibrator(
            lr=config.platt_lr,
            max_iter=config.platt_max_iter,
            tol=config.platt_tol,
            patience=config.platt_patience,
        )
        cal.fit(s_val, y_val)
        s_test = _logit(p_test)
        p_test_cal = cal.predict_proba(s_test)[:, 1]
        y_pred_cal = (p_test_cal >= config.threshold).astype(int)

        cal_metrics = {
            "accuracy": accuracy_score(y_test, y_pred_cal),
            "roc_auc": roc_auc_score(y_test, p_test_cal),
            "log_loss": log_loss(y_test, p_test_cal),
            "brier": brier_score_loss(y_test, p_test_cal),
        }
        platt_params = {"a": float(cal.a), "c": float(cal.c)}

    # “importance” proxy: |w| since we standardized features
    w = model.w.copy()
    abs_w = np.abs(w)
    top_idx = np.argsort(-abs_w)[:10]
    top_weights = [
        {"feature": feature_names[i], "weight": float(w[i]), "abs_weight": float(abs_w[i])}
        for i in top_idx
    ]

    run = {
        "run_id": _utc_timestamp(),
        "dataset": {
            "name": "sklearn.datasets.load_breast_cancer",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "feature_names": feature_names,
            "target_mean": float(np.mean(y)),
        },
        "config": asdict(config),
        "results": {
            "metrics": {k: _jsonable(v) for k, v in metrics.items()},
            "n_iters_ran": int(len(model.history_)),
            "history": [{"iter": int(it), "train_loss": float(tr), "val_loss": float(va)} for (it, tr, va) in model.history_],
            "weights": _jsonable(w),
            "bias": float(model.b),
            "top_weights": top_weights,
            "y_test": _jsonable(y_test),
            "p_test": _jsonable(p_test),

            # difficulty knobs outcome
            "data_stress": {
                "label_noise_rate": float(config.label_noise_rate),
                "n_labels_flipped_train": int(n_flipped),
                "feature_noise_std": float(config.feature_noise_std),
                "feature_noise_std_realized": float(realized_feature_noise),
            },

            # calibration block
            "calibration": (
                None
                if not config.use_platt
                else {
                    "method": "PlattScaling",
                    "params": platt_params,
                    "metrics": {k: _jsonable(v) for k, v in cal_metrics.items()},
                    "p_test_cal": _jsonable(p_test_cal),
                }
            ),

            # benchmark block
            "benchmark": (
                None
                if bench is None
                else {
                    "name": "RandomForestClassifier",
                    "notes": "Trained on raw features (train may include feature noise if enabled).",
                    "config": {
                        "n_estimators": int(config.rf_n_estimators),
                        "max_depth": config.rf_max_depth,
                        "min_samples_leaf": int(config.rf_min_samples_leaf),
                    },
                    "metrics": {k: _jsonable(v) for k, v in bench_metrics.items()},
                    "p_test": _jsonable(p_test_b),
                }
            ),
        },
        "notes": {
            "scaling": "StandardScaler fit on train only (after optional train feature noise); applied to val/test.",
            "interpretation": "Weights are comparable because features are standardized.",
            "stress": "Label/feature noise are train-only stressors to make improvements measurable.",
        },
    }

    fname = run["run_id"].replace(":", "").replace("-", "")
    out_file = runs_path / f"run_{fname}.json"
    out_file.write_text(json.dumps(run, indent=2))
    return out_file


if __name__ == "__main__":
    path = run_and_log(RunConfig())
    print(f"Saved run to: {path}")
