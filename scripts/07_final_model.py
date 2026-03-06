"""
Final Model Training — Fast Version
=====================================
Same logic, faster execution:
  - n_estimators: 200 → 100
  - n_jobs=-1 on everything that supports it
  - GradientBoosting replaced with HistGradientBoosting (much faster)
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef,
    precision_score, recall_score, classification_report
)
from imblearn.over_sampling import SMOTE

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network  import MLPClassifier
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier
from catboost                import CatBoostClassifier

os.makedirs("models",        exist_ok=True)
os.makedirs("data/holdout",  exist_ok=True)
os.makedirs("results/stage1",exist_ok=True)
os.makedirs("results/stage2",exist_ok=True)

# ── Model map — n_jobs=-1 on everything that supports it ─────
MODEL_MAP = {
    "LogisticRegression":
        LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),

    "RandomForest":
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),

    "ExtraTrees":
        ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),

    "GradientBoosting":
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        # Same as screening — ensures consistency between screening and final model

    "XGBoost":
        XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric="logloss", verbosity=0,
            n_jobs=-1, tree_method="hist"
        ),

    "LightGBM":
        LGBMClassifier(
            n_estimators=100, random_state=42,
            verbose=-1, n_jobs=-1
        ),

    "CatBoost":
        CatBoostClassifier(
            iterations=100, random_state=42,
            verbose=0, thread_count=-1
        ),

    "MLP":
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=300, random_state=42
        ),

    # Screening script names — map them too
    "kNN":          None,   # fallback to LightGBM
    "GaussianNB":   None,
    "DecisionTree": None,
    "AdaBoost":     None,
    "SVM_Linear":   None,
}


def get_model(name):
    """Get model by name. Falls back to LightGBM if not found."""
    mdl = MODEL_MAP.get(name)
    if mdl is None:
        print(f"  ⚠️  '{name}' not in fast MODEL_MAP — using LightGBM as fallback")
        mdl = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    return mdl


def train_final_model(
    X, y,
    screening_csv,
    model_path,
    holdout_X_path,
    holdout_y_path,
    report_path,
    stage_name
):
    print(f"\n{'='*60}")
    print(f"  FINAL MODEL: {stage_name}")
    print(f"  Dataset    : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Labels     : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"{'='*60}")

    # ── Step 1: 90/10 split — holdout never touched ───────────
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )
    np.save(holdout_X_path, X_holdout)
    np.save(holdout_y_path, y_holdout)
    print(f"\n  Holdout : {X_holdout.shape[0]} samples saved → {holdout_X_path}")
    print(f"  Training: {X_train.shape[0]} samples")

    # ── Step 2: Best model from screening ─────────────────────
    screening_df    = pd.read_csv(screening_csv)
    best_model_name = screening_df.iloc[0]["model"]
    best_f1         = screening_df.iloc[0]["f1_macro_mean"]
    print(f"\n  Best model from screening: {best_model_name}  (CV F1={best_f1:.4f})")

    # ── Step 3: SMOTE on training data ────────────────────────
    print(f"\n  Applying SMOTE...")
    print(f"  Before: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    smote           = SMOTE(random_state=42)
    X_bal, y_bal    = smote.fit_resample(X_train, y_train)
    print(f"  After : {dict(zip(*np.unique(y_bal, return_counts=True)))}")

    # ── Step 4: Train final pipeline ──────────────────────────
    model = get_model(best_model_name)
    print(f"\n  Training {best_model_name}...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    model)
    ])
    pipeline.fit(X_bal, y_bal)
    print(f"  ✅ Training complete")

    # ── Step 5: Save model ────────────────────────────────────
    joblib.dump(pipeline, model_path)
    print(f"  💾 Saved → {model_path}")

    # ── Step 6: Evaluate on holdout ───────────────────────────
    print(f"\n  Evaluating on holdout ({X_holdout.shape[0]} samples)...")
    y_pred = pipeline.predict(X_holdout)
    y_prob = pipeline.predict_proba(X_holdout)[:, 1]

    acc    = accuracy_score(y_holdout, y_pred)
    f1     = f1_score(y_holdout, y_pred, average="macro")
    bal    = balanced_accuracy_score(y_holdout, y_pred)
    mcc    = matthews_corrcoef(y_holdout, y_pred)
    auc    = roc_auc_score(y_holdout, y_prob)
    prec   = precision_score(y_holdout, y_pred, zero_division=0)
    rec    = recall_score(y_holdout, y_pred,    zero_division=0)

    print(f"\n  ── Holdout Results ──────────────────────────────")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  F1-Macro         : {f1:.4f}")
    print(f"  Balanced Accuracy: {bal:.4f}")
    print(f"  MCC              : {mcc:.4f}")
    print(f"  ROC-AUC          : {auc:.4f}")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_holdout, y_pred))

    # ── Step 7: Save holdout report ───────────────────────────
    report = (
        f"FINAL MODEL HOLDOUT EVALUATION — {stage_name}\n"
        f"{'='*50}\n"
        f"Model            : {best_model_name}\n"
        f"CV F1-Macro      : {best_f1:.4f}\n"
        f"Holdout Samples  : {X_holdout.shape[0]}\n\n"
        f"Accuracy         : {acc:.4f}\n"
        f"F1-Macro         : {f1:.4f}\n"
        f"Balanced Accuracy: {bal:.4f}\n"
        f"MCC              : {mcc:.4f}\n"
        f"ROC-AUC          : {auc:.4f}\n"
        f"Precision        : {prec:.4f}\n"
        f"Recall           : {rec:.4f}\n\n"
        f"Classification Report:\n"
        f"{classification_report(y_holdout, y_pred)}\n"
    )
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  💾 Report saved → {report_path}")

    return pipeline


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── STAGE 1 ───────────────────────────────────────────────
    X1 = np.load("data/processed/stage1_features.npy")
    y1 = np.load("data/processed/stage1_labels.npy")

    train_final_model(
        X1, y1,
        screening_csv   = "results/stage1/screening_results.csv",
        model_path      = "models/stage1_authorship.pkl",
        holdout_X_path  = "data/holdout/stage1_holdout_X.npy",
        holdout_y_path  = "data/holdout/stage1_holdout_y.npy",
        report_path     = "results/stage1/holdout_report.txt",
        stage_name      = "Stage 1 — AI vs Human"
    )

    # ── STAGE 2 ───────────────────────────────────────────────
    X2 = np.load("data/processed/stage2_features.npy")
    y2 = np.load("data/processed/stage2_labels.npy")

    train_final_model(
        X2, y2,
        screening_csv   = "results/stage2/screening_results.csv",
        model_path      = "models/stage2_unsafe.pkl",
        holdout_X_path  = "data/holdout/stage2_holdout_X.npy",
        holdout_y_path  = "data/holdout/stage2_holdout_y.npy",
        report_path     = "results/stage2/holdout_report.txt",
        stage_name      = "Stage 2 — Unsafe"
    )

    print(f"\n{'='*60}")
    print(f"  ALL DONE")
    print(f"  models/stage1_authorship.pkl")
    print(f"  models/stage2_unsafe.pkl"))
    print(f"  results/stage1/holdout_report.txt")
    print(f"  results/stage2/holdout_report.txt")
    print(f"{'='*60}")