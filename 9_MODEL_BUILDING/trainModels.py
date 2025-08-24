# 9_MODEL_BUILDING/train_from_feature_store.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import sys, json, warnings, time, sqlite3
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FS_DB       = PROJECT_ROOT / "6_DATA_TRANSFORMATION_AND_STORAGE" / "feature_store.db"
FS_API_DIR  = PROJECT_ROOT / "7_FEATURE_STORE"      # featureApi.py lives here
OUT_DIR     = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Config knobs ----------------
RANDOM_SEED  = 42
ID_COL       = "customer_id"
TARGET_COL   = "churned"
EXCLUDE_COLS = {ID_COL, TARGET_COL, "__source__", "signup_date", "created_at", "registration_date"}

# SPEED toggles
QUICK      = True     # << set False for a fuller sweep
ENABLE_XGB = False    # turn on if xgboost is installed and you want it

# Optional sub-sampling fraction used when QUICK is True
QUICK_SAMPLE_FRAC = 0.35   # ~35% of data for a smoke test

# -------------------------------------------------
def ts_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ---------------- Data loading ----------------
def load_from_api_or_sql() -> pd.DataFrame:
    """
    Preferred: use your Feature Store API (7_FEATURE_STORE/featureApi.py).
    Fallback: read the entire 'feature_store' table via SQLite.
    """
    # Try API first
    try:
        sys.path.insert(0, str(FS_API_DIR))
        from featureApi import get_training_dataframe  # your API helper
        df = get_training_dataframe()
        if isinstance(df, pd.DataFrame) and not df.empty:
            print("[INFO] Loaded training dataframe from featureApi.get_training_dataframe()")
            return df
    except Exception as e:
        print("[WARN] API load failed, falling back to SQLite:", e)

    # Fallback to SQLite
    if not FS_DB.exists():
        raise FileNotFoundError(f"Feature store DB not found: {FS_DB}")
    with sqlite3.connect(str(FS_DB)) as con:
        df = pd.read_sql_query("SELECT * FROM feature_store", con)
    print(f"[INFO] Loaded {len(df):,} rows from SQLite feature_store table")
    return df

def split_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df[TARGET_COL].astype(int).values
    return X, y, list(X.columns)

# ---------------- Training ----------------
def train_and_compare(X: pd.DataFrame, y: np.ndarray):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )

    models = []

    # Logistic Regression (regularized, balanced)
    logit = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=(800 if QUICK else 1000),
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=1,
            random_state=RANDOM_SEED
        ))
    ])
    logit_grid = {"clf__C": ([1.0] if QUICK else [0.1, 1.0, 3.0])}
    models.append(("logreg", logit, logit_grid))

    # Random Forest (balanced, constrained to reduce overfit)
    rf = RandomForestClassifier(
        n_estimators=(120 if QUICK else 400),
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    rf_grid = {
        "max_depth": ([None] if QUICK else [None, 10, 16]),
        "min_samples_leaf": ([1] if QUICK else [1, 3]),
    }
    models.append(("rf", rf, rf_grid))

    # Optional XGBoost
    if ENABLE_XGB:
        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(
                n_estimators=(200 if QUICK else 500),
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                eval_metric="logloss",
            )
            models.append(("xgb", xgb, {"max_depth": ([6] if QUICK else [4, 6, 8])}))
        except Exception as e:
            print("[WARN] XGBoost not available:", e)

    # Cross-validation folds (fewer in QUICK)
    cv = StratifiedKFold(n_splits=(3 if QUICK else 5), shuffle=True, random_state=RANDOM_SEED)

    results = []
    for name, est, grid in models:
        print(f"\n[INFO] >>> Training model: {name}")
        start = time.time()
        try:
            gs = GridSearchCV(
                est, grid, cv=cv, scoring="f1", n_jobs=-1, refit=True, verbose=2
            )
            gs.fit(X_tr, y_tr)
            y_pred = gs.predict(X_te)
            y_prob = gs.predict_proba(X_te)[:, 1] if hasattr(gs.best_estimator_, "predict_proba") else None
            elapsed = time.time() - start

            metrics = {
                "model": name,
                "best_params": gs.best_params_,
                "accuracy": float(accuracy_score(y_te, y_pred)),
                "precision": float(precision_score(y_te, y_pred, zero_division=0)),
                "recall": float(recall_score(y_te, y_pred, zero_division=0)),
                "f1": float(f1_score(y_te, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_te, y_prob)) if y_prob is not None else None,
                "train_seconds": round(elapsed, 2),
            }
            print(f"[INFO] <<< Done {name} | best={gs.best_params_} | "
                  f"F1={metrics['f1']:.3f} AUC={metrics['roc_auc']}")
            results.append((name, gs.best_estimator_, (X_te, y_te, y_pred, y_prob), metrics))
        except Exception as e:
            print(f"[WARN] {name} failed:", e)

    if not results:
        raise RuntimeError("No model trained successfully.")
    # pick by F1 then ROC-AUC
    results.sort(key=lambda r: (r[3]["f1"], r[3].get("roc_auc") or 0.0), reverse=True)
    return results[0], results

# ---------------- Outputs ----------------
def save_plots(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center")
    fig.tight_layout()
    cm_png = OUT_DIR / "confusion_matrix.png"
    fig.savefig(cm_png, dpi=200); plt.close(fig)

    roc_png = None
    if y_prob is not None and len(np.unique(y_true)) == 2:
        fig2, ax2 = plt.subplots()
        RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax2, name="ROC")
        fig2.tight_layout()
        roc_png = OUT_DIR / "roc_curve.png"
        fig2.savefig(roc_png, dpi=200); plt.close(fig2)
    return cm_png, roc_png

def save_artifacts(best_name, best_estimator, test_bundle, metrics, feature_names):
    X_te, y_te, y_pred, y_prob = test_bundle

    cm_png, roc_png = save_plots(y_te, y_pred, y_prob)

    rep_txt = OUT_DIR / "classification_report.txt"
    with open(rep_txt, "w", encoding="utf-8") as f:
        f.write(classification_report(y_te, y_pred, digits=4))

    metrics_full = dict(metrics)
    metrics_full.update({
        "created_at_utc": ts_utc(),
        "n_features": len(feature_names),
        "features": feature_names,
        "artifacts": {
            "confusion_matrix_png": str(cm_png),
            "roc_curve_png": str(roc_png) if roc_png else None,
            "classification_report_txt": str(rep_txt),
        }
    })
    metrics_json = OUT_DIR / "metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_full, f, indent=2)

    model_pkl = OUT_DIR / "best_model.pkl"
    joblib.dump({"model": best_estimator, "features": feature_names}, model_pkl)

    manifest = {
        "saved_at_utc": ts_utc(),
        "best_model": best_name,
        "model_file": str(model_pkl),
        "metrics_file": str(metrics_json),
        "data_source": str(FS_DB),
        "quick_mode": QUICK,
    }
    with open(OUT_DIR / "model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nArtifacts saved:")
    for p in [model_pkl, metrics_json, rep_txt, cm_png, roc_png]:
        if p: print(" -", Path(p).name)

    # Optional MLflow logging
    try:
        import mlflow
        from mlflow.models.signature import infer_signature
        mlflow.set_experiment("dmml_churn")
        with mlflow.start_run(run_name=f"best::{best_name}"):
            for k, v in (metrics.get("best_params") or {}).items():
                mlflow.log_param(k, v)
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "train_seconds"]:
                if metrics.get(k) is not None:
                    mlflow.log_metric(k, float(metrics[k]))
            mlflow.log_artifact(str(metrics_json))
            mlflow.log_artifact(str(rep_txt))
            mlflow.log_artifact(str(cm_png))
            if roc_png: mlflow.log_artifact(str(roc_png))
            sig = infer_signature(X_te, y_pred)
            mlflow.sklearn.log_model(best_estimator, "model", signature=sig)
        print("[MLflow] run logged.")
    except Exception as e:
        print("[INFO] MLflow not used (install mlflow to enable).", e)

# ---------------- Main ----------------
def main():
    warnings.filterwarnings("ignore")

    # 1) Load data
    df = load_from_api_or_sql()
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")
    X, y, features = split_xy(df)
    print(f"[INFO] X shape={X.shape} | churn rate={y.mean():.3f}")

    # 2) Optional quick sampling (keeps class balance)
    if QUICK and QUICK_SAMPLE_FRAC < 1.0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-QUICK_SAMPLE_FRAC, random_state=RANDOM_SEED)
        (idx, _), = sss.split(X, y)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]
        print(f"[INFO] QUICK sample -> X shape={X.shape}")

    # 3) Train & compare
    (best_name, best_est, test_bundle, metrics), _ = train_and_compare(X, y)
    print("[INFO] Best model:", best_name, "| metrics:", metrics)

    # 4) Save artifacts
    save_artifacts(best_name, best_est, test_bundle, metrics, features)
    print("\n[DONE] Training from feature store complete.")

if __name__ == "__main__":
    main()
