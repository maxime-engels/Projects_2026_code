import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from src.experiment import RunConfig, run_and_log

RUNS_DIR = Path("runs")


def load_runs() -> list[dict]:
    """
    Load all run JSON artifacts in chronological order (oldest -> newest),
    so the history grows downward as you add runs.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RUNS_DIR.glob("run_*.json"), reverse=False)  # oldest -> newest
    runs = []
    for f in files:
        try:
            runs.append(json.loads(f.read_text()))
        except Exception:
            continue
    return runs


def runs_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = r["results"]["metrics"]
        cfg = r["config"]
        rows.append(
            {
                "run_id": r["run_id"],
                "accuracy": float(m["accuracy"]),
                "roc_auc": float(m["roc_auc"]),
                "log_loss": float(m["log_loss"]),
                "brier": float(m["brier"]),
                "lr": float(cfg["lr"]),
                "reg_lambda": float(cfg["reg_lambda"]),
                "max_iter": int(cfg["max_iter"]),
                "patience": int(cfg["patience"]),
                "seed": int(cfg["seed"]),
                "label_noise": float(cfg.get("label_noise_rate", 0.0)),
                "feat_noise": float(cfg.get("feature_noise_std", 0.0)),
                "platt": bool(cfg.get("use_platt", False)),
                "benchmark": bool(cfg.get("run_benchmark", False)),
            }
        )
    return pd.DataFrame(rows)


def interpret_metrics(m: dict) -> dict:
    roc = float(m["roc_auc"])
    ll = float(m["log_loss"])
    br = float(m["brier"])
    acc = float(m["accuracy"])

    if roc >= 0.98:
        discr = "Excellent discrimination (very strong ranking)."
    elif roc >= 0.95:
        discr = "Strong discrimination."
    elif roc >= 0.90:
        discr = "Moderate discrimination."
    else:
        discr = "Weak discrimination (ranking quality is limited)."

    if ll <= 0.12 and br <= 0.06:
        probq = "Probabilities look strong (low log loss and low Brier)."
    elif ll <= 0.20 and br <= 0.09:
        probq = "Probabilities look decent; calibration may still be improvable."
    else:
        probq = "Probabilities may be miscalibrated or overconfident; prioritize calibration diagnostics and calibration methods."

    if roc >= 0.95 and ll <= 0.20:
        conf = "High confidence in ranking; moderate confidence in probability quality."
    elif roc >= 0.90:
        conf = "Moderate confidence; focus on stability and calibration."
    else:
        conf = "Low confidence; revisit assumptions or add stress tests and stronger baselines."

    return {
        "discrimination": discr,
        "probabilities": probq,
        "confidence": conf,
        "accuracy_note": f"Accuracy={acc:.3f} is threshold-dependent.",
    }


def derive_next_steps(run: dict, prev_run: dict | None = None) -> list[str]:
    """
    Conservative, action-oriented suggestions derived from run artifacts.
    """
    sugg: list[str] = []
    cfg = run["config"]
    hist = run["results"]["history"]
    m = run["results"]["metrics"]

    # Convergence / plateau
    n_ran = int(run["results"]["n_iters_ran"])
    if n_ran >= int(cfg["max_iter"]):
        sugg.append("Reached max_iter: likely not fully converged. Try a slightly higher lr or reduce reg_lambda slightly.")
    else:
        if len(hist) >= 200:
            early = hist[max(0, len(hist) - 200)]
            if float(early["val_loss"]) - float(hist[-1]["val_loss"]) < 0.01:
                sugg.append("Validation loss barely improved in the last ~200 steps: change lr or reg_lambda, not max_iter.")

    # Overfit/underfit
    if hist:
        train_loss = float(hist[-1]["train_loss"])
        val_loss = float(hist[-1]["val_loss"])
        gap = val_loss - train_loss
        if gap > 0.05:
            sugg.append("Train–val gap suggests overfitting: increase reg_lambda (x2–x5) or reduce patience.")
        elif abs(gap) <= 0.01 and val_loss > 0.20:
            sugg.append("Train and val loss are close but not low: underfitting; reduce reg_lambda or improve optimizer (later).")

    # Overconfidence hint
    p = np.array(run["results"]["p_test"], dtype=float)
    extreme_frac = float(np.mean((p < 0.02) | (p > 0.98)))
    if extreme_frac > 0.60 and float(m["brier"]) > 0.07:
        sugg.append("Predictions are very extreme while Brier is not low: enable Platt scaling and evaluate ECE/Brier changes.")

    # If plateau vs previous
    if prev_run is not None:
        mp = prev_run["results"]["metrics"]
        d_auc = float(m["roc_auc"]) - float(mp["roc_auc"])
        d_ll = float(m["log_loss"]) - float(mp["log_loss"])
        if abs(d_auc) < 0.002 and abs(d_ll) < 0.01:
            sugg.append("Plateau vs previous: turn on a controlled stressor (label_noise_rate=0.05) to make improvements measurable, or compare against RandomForest.")
        if d_auc < -0.01 and d_ll > 0.02:
            sugg.append("Regression vs previous: revert one change and isolate variables.")

    sugg.append("Run discipline: change exactly one knob per run and use deltas on Home to interpret impact.")
    out = []
    seen = set()
    for s in sugg:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def calibration_report(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)

        cnt = int(mask.sum())
        if cnt == 0:
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "count": 0, "avg_conf": np.nan, "frac_pos": np.nan, "gap": np.nan})
            continue

        avg_conf = float(p[mask].mean())
        frac_pos = float(y[mask].mean())
        gap = abs(avg_conf - frac_pos)
        ece += (cnt / len(p)) * gap
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "count": cnt, "avg_conf": avg_conf, "frac_pos": frac_pos, "gap": gap})

    return float(ece), pd.DataFrame(rows)


def plot_reliability(df_rel: pd.DataFrame, title: str):
    d = df_rel.dropna(subset=["avg_conf", "frac_pos"])
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", label="ideal")
    ax.plot(d["avg_conf"], d["frac_pos"], marker="o", label="model")
    ax.set_title(title)
    ax.set_xlabel("avg predicted probability (bin)")
    ax.set_ylabel("empirical positive rate (bin)")
    ax.legend()
    st.pyplot(fig)


def plot_class_balance(y: np.ndarray):
    counts = np.bincount(y.astype(int), minlength=2)
    fig, ax = plt.subplots()
    ax.bar(["class 0", "class 1"], counts)
    ax.set_title("Class balance")
    ax.set_ylabel("count")
    st.pyplot(fig)


def plot_feature_hist(X: np.ndarray, feature_names: list[str], idx: int):
    fig, ax = plt.subplots()
    ax.hist(X[:, idx], bins=30)
    ax.set_title(f"Feature distribution: {feature_names[idx]}")
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    st.pyplot(fig)


def plot_corr_heatmap(X: np.ndarray, feature_names: list[str], max_features: int = 20):
    variances = np.var(X, axis=0)
    sel = np.argsort(-variances)[:max_features]
    Xs = X[:, sel]
    names = [feature_names[i] for i in sel]
    corr = np.corrcoef(Xs, rowvar=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, aspect="auto")
    ax.set_title(f"Correlation heatmap (top {max_features} variance features)")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


def plot_loss_curve(run: dict):
    hist = run["results"]["history"]
    it = [h["iter"] for h in hist]
    tr = [h["train_loss"] for h in hist]
    va = [h["val_loss"] for h in hist]

    fig, ax = plt.subplots()
    ax.plot(it, tr, label="train_loss")
    ax.plot(it, va, label="val_loss")
    ax.set_title("Training dynamics: loss vs iteration")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend()
    st.pyplot(fig)


def plot_pred_hist(p: np.ndarray, title: str):
    fig, ax = plt.subplots()
    ax.hist(p, bins=30)
    ax.set_title(title)
    ax.set_xlabel("p(y=1)")
    ax.set_ylabel("count")
    st.pyplot(fig)


def plot_top_weights(run: dict, top_k: int = 10):
    tw = run["results"]["top_weights"][:top_k]
    feats = [x["feature"] for x in tw]
    vals = [x["weight"] for x in tw]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feats[::-1], vals[::-1])
    ax.set_title(f"Top {top_k} weights (standardized features)")
    ax.set_xlabel("weight")
    st.pyplot(fig)


def plot_metric_trends(df: pd.DataFrame):
    if df.empty:
        return
    fig, ax = plt.subplots()
    ax.plot(df["run_id"], df["roc_auc"], label="roc_auc")
    ax.plot(df["run_id"], df["log_loss"], label="log_loss")
    ax.plot(df["run_id"], df["brier"], label="brier")
    ax.set_title("Trends across runs (chronological)")
    ax.set_xlabel("run_id (UTC)")
    ax.set_ylabel("metric value")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    st.pyplot(fig)


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="ML Lab: Data + Runs", layout="wide")
st.title("ML Lab: Understand the Data, Track Runs, Learn ML")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Dataset overview", "EDA", "Run & log", "Run history", "Run details", "Compare runs"],
)

# Load dataset for EDA pages
data = load_breast_cancer()
X = data["data"].astype(float)
y = data["target"].astype(int)
feature_names = list(data["feature_names"])


if page == "Home":
    st.subheader("Main goal")
    st.write(
        "Build a rigorous ML lab around a from-scratch logistic regression: "
        "understand the dataset, understand training dynamics, and track improvements across runs via reproducible logs."
    )

    runs = load_runs()
    if not runs:
        st.info("No runs yet. Go to 'Run & log' to create your first run.")
    else:
        latest = runs[-1]
        prev = runs[-2] if len(runs) >= 2 else None

        st.subheader("Current status (latest run)")
        m = latest["results"]["metrics"]
        interp = interpret_metrics(m)

        c1, c2, c3, c4 = st.columns(4)
        if prev is not None:
            mp = prev["results"]["metrics"]
            c1.metric("ROC-AUC", f"{m['roc_auc']:.4f}", f"{(float(m['roc_auc'])-float(mp['roc_auc'])):+.4f}")
            c2.metric("Log loss", f"{m['log_loss']:.4f}", f"{(float(m['log_loss'])-float(mp['log_loss'])):+.4f}")
            c3.metric("Brier", f"{m['brier']:.4f}", f"{(float(m['brier'])-float(mp['brier'])):+.4f}")
            c4.metric("Accuracy", f"{m['accuracy']:.4f}", f"{(float(m['accuracy'])-float(mp['accuracy'])):+.4f}")
        else:
            c1.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
            c2.metric("Log loss", f"{m['log_loss']:.4f}")
            c3.metric("Brier", f"{m['brier']:.4f}")
            c4.metric("Accuracy", f"{m['accuracy']:.4f}")

        # Benchmark summary if present
        bench = latest["results"].get("benchmark")
        if bench is not None:
            st.subheader("Benchmark (latest run)")
            bm = bench["metrics"]
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("RF ROC-AUC", f"{float(bm['roc_auc']):.4f}")
            b2.metric("RF Log loss", f"{float(bm['log_loss']):.4f}")
            b3.metric("RF Brier", f"{float(bm['brier']):.4f}")
            b4.metric("RF Accuracy", f"{float(bm['accuracy']):.4f}")

        # Calibration summary
        st.subheader("Probability calibration (latest run — compact)")
        p = np.array(latest["results"]["p_test"], dtype=float)
        yt = np.array(latest["results"]["y_test"], dtype=int)
        ece, _ = calibration_report(p, yt, n_bins=10)
        st.metric("ECE (from-scratch)", f"{ece:.4f}")

        cal = latest["results"].get("calibration")
        if cal is not None:
            p_cal = np.array(cal["p_test_cal"], dtype=float)
            ece_cal, _ = calibration_report(p_cal, yt, n_bins=10)
            st.metric("ECE (Platt-calibrated)", f"{ece_cal:.4f}")

        if bench is not None:
            p_b = np.array(bench["p_test"], dtype=float)
            ece_b, _ = calibration_report(p_b, yt, n_bins=10)
            st.metric("ECE (benchmark)", f"{ece_b:.4f}")

        st.subheader("What the latest results mean (concise)")
        st.write(f"- {interp['discrimination']}")
        st.write(f"- {interp['probabilities']}")
        st.write(f"- {interp['confidence']}")
        st.caption(interp["accuracy_note"])

        st.subheader("Trends across runs (chronological)")
        df = runs_table(runs)
        plot_metric_trends(df)

        st.subheader("Recommended next experiments (based on latest run)")
        suggestions = derive_next_steps(latest, prev_run=prev)
        for s in suggestions:
            st.write(f"- {s}")

    st.divider()
    st.write("Use the sidebar to explore the dataset (EDA), create new runs, and compare runs in detail.")


elif page == "Dataset overview":
    st.subheader("What you are looking at")
    st.write(
        "Breast Cancer Wisconsin dataset: each row is a tumor sample, "
        "each column is a numeric feature computed from an image, target is binary (0/1)."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Samples", X.shape[0])
        st.metric("Features", X.shape[1])
    with col2:
        st.metric("Positive rate (mean y)", float(np.mean(y)))
        st.metric("Negatives", int(np.sum(y == 0)))

    st.subheader("Feature list (first 15)")
    st.write(pd.DataFrame({"feature": feature_names}).head(15))

    st.subheader("Class balance")
    plot_class_balance(y)


elif page == "EDA":
    st.subheader("Exploratory Data Analysis")
    st.write("Goal: make the dataset visually understandable before changing modeling choices.")

    col1, col2 = st.columns(2)
    with col1:
        idx = st.selectbox(
            "Pick a feature to visualize",
            options=list(range(len(feature_names))),
            format_func=lambda i: feature_names[i],
        )
        plot_feature_hist(X, feature_names, idx)
    with col2:
        maxf = st.slider("Correlation heatmap size", 10, 40, 20, 5)
        plot_corr_heatmap(X, feature_names, max_features=maxf)


elif page == "Run & log":
    st.subheader("Run an experiment and log it (with timestamp)")
    st.write("This trains your from-scratch logistic regression and saves a JSON artifact in ./runs/.")

    # --- Session-state defaults (persist across runs) ---
    defaults = {
        "lr": 0.2,
        "reg_lambda": 0.01,
        "max_iter": 20000,
        "patience": 50,
        "seed": 42,
        "threshold": 0.50,
        "label_noise_rate": 0.0,
        "feature_noise_std": 0.0,
        "use_platt": False,
        "run_benchmark": True,
        "rf_n_estimators": 400,
        "rf_max_depth": None,
        "rf_min_samples_leaf": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # --- Suggested next tweak (based on latest run) ---
    runs = load_runs()
    latest = runs[-1] if runs else None
    prev = runs[-2] if runs and len(runs) >= 2 else None

    with st.expander("Suggested next tweak (based on latest run)"):
        if latest is None:
            st.write("No runs yet. Create your first run to receive suggestions.")
        else:
            suggestions = derive_next_steps(latest, prev_run=prev)
            for s in suggestions:
                st.write(f"- {s}")

            st.write("Apply one tweak (updates form defaults):")
            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                if st.button("reg_lambda x2"):
                    st.session_state.reg_lambda = float(st.session_state.reg_lambda) * 2.0
            with c2:
                if st.button("reg_lambda /2"):
                    st.session_state.reg_lambda = max(0.0, float(st.session_state.reg_lambda) / 2.0)
            with c3:
                if st.button("lr x1.2"):
                    st.session_state.lr = float(st.session_state.lr) * 1.2
            with c4:
                if st.button("Enable Platt"):
                    st.session_state.use_platt = True
            with c5:
                if st.button("Stress: label noise 5%"):
                    st.session_state.label_noise_rate = 0.05

            st.caption("These buttons only set defaults. The run config is always logged; nothing is hidden.")

    # --- Run form ---
    with st.form("run_form"):
        st.write("### Core knobs (from-scratch logistic)")
        lr = st.number_input("learning rate", value=float(st.session_state.lr), min_value=1e-4, max_value=10.0, format="%.4f")
        reg_lambda = st.number_input("L2 lambda", value=float(st.session_state.reg_lambda), min_value=0.0, max_value=10.0, format="%.6f")
        max_iter = st.number_input("max iterations", value=int(st.session_state.max_iter), min_value=100, max_value=500000, step=1000)
        patience = st.number_input("early-stop patience", value=int(st.session_state.patience), min_value=5, max_value=2000, step=5)
        seed = st.number_input("random seed", value=int(st.session_state.seed), min_value=0, max_value=10_000)
        threshold = st.number_input("classification threshold", value=float(st.session_state.threshold), min_value=0.0, max_value=1.0, format="%.2f")

        with st.expander("Advanced knobs (stress tests, calibration, benchmark)"):
            st.write("#### Controlled difficulty (train-only)")
            label_noise_rate = st.slider("label_noise_rate (flip % train labels)", 0.0, 0.20, float(st.session_state.label_noise_rate), 0.01)
            feature_noise_std = st.slider("feature_noise_std (Gaussian noise on train features)", 0.0, 2.0, float(st.session_state.feature_noise_std), 0.05)

            st.write("#### Calibration")
            use_platt = st.checkbox("Use Platt scaling (fit on validation)", value=bool(st.session_state.use_platt))

            st.write("#### Benchmark (RandomForest)")
            run_benchmark = st.checkbox("Run benchmark", value=bool(st.session_state.run_benchmark))
            rf_n_estimators = st.number_input("RF n_estimators", value=int(st.session_state.rf_n_estimators), min_value=50, max_value=2000, step=50)
            rf_min_samples_leaf = st.number_input("RF min_samples_leaf", value=int(st.session_state.rf_min_samples_leaf), min_value=1, max_value=20, step=1)
            rf_max_depth_raw = st.text_input("RF max_depth (blank = None)", value="" if st.session_state.rf_max_depth is None else str(st.session_state.rf_max_depth))

        submitted = st.form_submit_button("Run experiment")

    if submitted:
        # Persist chosen values as new defaults
        st.session_state.lr = float(lr)
        st.session_state.reg_lambda = float(reg_lambda)
        st.session_state.max_iter = int(max_iter)
        st.session_state.patience = int(patience)
        st.session_state.seed = int(seed)
        st.session_state.threshold = float(threshold)

        st.session_state.label_noise_rate = float(label_noise_rate)
        st.session_state.feature_noise_std = float(feature_noise_std)
        st.session_state.use_platt = bool(use_platt)

        st.session_state.run_benchmark = bool(run_benchmark)
        st.session_state.rf_n_estimators = int(rf_n_estimators)
        st.session_state.rf_min_samples_leaf = int(rf_min_samples_leaf)

        rf_max_depth = None
        if rf_max_depth_raw.strip() != "":
            try:
                rf_max_depth = int(rf_max_depth_raw.strip())
            except ValueError:
                st.error("RF max_depth must be an integer or blank.")
                st.stop()
        st.session_state.rf_max_depth = rf_max_depth

        cfg = RunConfig(
            lr=float(lr),
            reg_lambda=float(reg_lambda),
            max_iter=int(max_iter),
            patience=int(patience),
            seed=int(seed),
            threshold=float(threshold),
            label_noise_rate=float(label_noise_rate),
            feature_noise_std=float(feature_noise_std),
            use_platt=bool(use_platt),
            run_benchmark=bool(run_benchmark),
            rf_n_estimators=int(rf_n_estimators),
            rf_min_samples_leaf=int(rf_min_samples_leaf),
            rf_max_depth=rf_max_depth,
        )

        out = run_and_log(cfg, runs_dir=str(RUNS_DIR))
        st.success(f"Saved run: {out.name}")

        r = json.loads(Path(out).read_text())
        st.subheader("Run results (from-scratch)")
        st.json(r["results"]["metrics"])

        if r["results"].get("calibration") is not None:
            st.subheader("Run results (Platt-calibrated)")
            st.json(r["results"]["calibration"]["metrics"])

        if r["results"].get("benchmark") is not None:
            st.subheader("Run results (benchmark)")
            st.json(r["results"]["benchmark"]["metrics"])

        st.caption("Go to Home for deltas and suggested next experiments. Go to Run details for calibration plots.")


elif page == "Run history":
    st.subheader("Run history (chronological, grows downward)")
    runs = load_runs()
    if not runs:
        st.info("No runs logged yet. Go to 'Run & log' first.")
    else:
        df = runs_table(runs)
        st.dataframe(df, use_container_width=True)


elif page == "Run details":
    st.subheader("Inspect one run deeply")
    runs = load_runs()
    if not runs:
        st.info("No runs logged yet. Go to 'Run & log' first.")
    else:
        ids = [r["run_id"] for r in runs]
        chosen = st.selectbox("Select run", options=ids, index=len(ids) - 1)
        run = next(r for r in runs if r["run_id"] == chosen)

        left, right = st.columns([1, 1])

        with left:
            st.write("**What you see**: training dynamics, test probability distribution, and calibration quality.")
            plot_loss_curve(run)

            p = np.array(run["results"]["p_test"], dtype=float)
            yt = np.array(run["results"]["y_test"], dtype=int)
            plot_pred_hist(p, "Predicted probabilities (test) — from-scratch")

            st.subheader("Calibration diagnostics (test set) — from-scratch")
            ece, rel = calibration_report(p, yt, n_bins=10)
            st.write(f"**ECE (from-scratch):** {ece:.4f}")
            plot_reliability(rel, "Reliability diagram — from-scratch")
            st.dataframe(rel, use_container_width=True)

            cal = run["results"].get("calibration")
            if cal is not None:
                p_cal = np.array(cal["p_test_cal"], dtype=float)
                plot_pred_hist(p_cal, "Predicted probabilities (test) — Platt-calibrated")

                st.subheader("Calibration diagnostics (test set) — Platt-calibrated")
                ece_cal, rel_cal = calibration_report(p_cal, yt, n_bins=10)
                st.write(f"**ECE (Platt-calibrated):** {ece_cal:.4f}")
                plot_reliability(rel_cal, "Reliability diagram — Platt-calibrated")
                st.dataframe(rel_cal, use_container_width=True)

            bench = run["results"].get("benchmark")
            if bench is not None:
                p_b = np.array(bench["p_test"], dtype=float)
                plot_pred_hist(p_b, "Predicted probabilities (test) — benchmark")

                st.subheader("Calibration diagnostics (test set) — benchmark")
                ece_b, rel_b = calibration_report(p_b, yt, n_bins=10)
                st.write(f"**ECE (benchmark):** {ece_b:.4f}")
                plot_reliability(rel_b, "Reliability diagram — benchmark")
                st.dataframe(rel_b, use_container_width=True)

        with right:
            st.write("**From-scratch metrics**")
            st.json(run["results"]["metrics"])

            cal = run["results"].get("calibration")
            if cal is not None:
                st.write("**Platt-calibrated metrics**")
                st.json(cal["metrics"])
                st.caption(f"Platt params: a={cal['params']['a']:.4f}, c={cal['params']['c']:.4f}")

            bench = run["results"].get("benchmark")
            if bench is not None:
                st.write("**Benchmark metrics**")
                st.json(bench["metrics"])
                st.caption(f"RF: n_estimators={bench['config']['n_estimators']}, min_samples_leaf={bench['config']['min_samples_leaf']}, max_depth={bench['config']['max_depth']}")

            st.write("**Top weights (from-scratch logistic)**")
            plot_top_weights(run, top_k=10)

            stress = run["results"].get("data_stress")
            if stress is not None:
                st.write("**Data stress (train-only)**")
                st.json(stress)

        with st.expander("Raw run JSON"):
            st.json(run)


elif page == "Compare runs":
    st.subheader("Compare two runs (chronological)")
    runs = load_runs()
    if len(runs) < 2:
        st.info("Need at least two runs to compare. Go to 'Run & log'.")
    else:
        ids = [r["run_id"] for r in runs]
        a = st.selectbox("Run A", options=ids, index=max(0, len(ids) - 2))
        b = st.selectbox("Run B", options=ids, index=len(ids) - 1)

        run_a = next(r for r in runs if r["run_id"] == a)
        run_b = next(r for r in runs if r["run_id"] == b)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Run A metrics")
            st.json(run_a["results"]["metrics"])
            plot_loss_curve(run_a)
        with col2:
            st.write("Run B metrics")
            st.json(run_b["results"]["metrics"])
            plot_loss_curve(run_b)

        st.write("Top weights comparison (A then B)")
        c1, c2 = st.columns(2)
        with c1:
            plot_top_weights(run_a)
        with c2:
            plot_top_weights(run_b)
