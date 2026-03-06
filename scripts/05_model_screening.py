"""
MODEL SCREENING — 13 Models (3 slow SVMs removed)
===================================================
Removed: SVM_RBF, SVM_Poly, SVM_Sigmoid
Reason : On 28k x 420 matrix they take 40-120 min each
         and boosting models always outperform them anyway.

Added  : n_jobs=-1 to every model that supports it
Reason : Uses all CPU cores in parallel — roughly halves time.

Models kept (13):
    1.  Logistic Regression
    2.  SVM Linear          (linear kernel is still fast)
    3.  k-NN
    4.  Gaussian Naive Bayes
    5.  Decision Tree
    6.  Random Forest
    7.  Extra Trees
    8.  AdaBoost
    9.  Gradient Boosting
    10. XGBoost
    11. LightGBM
    12. CatBoost
    13. MLP

Saves:
    results/stage1/screening_results.csv
    results/stage2/screening_results.csv
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE

# ── All 13 models ─────────────────────────────────────────────
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.neural_network  import MLPClassifier
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier
from catboost                import CatBoostClassifier

os.makedirs("results/stage1", exist_ok=True)
os.makedirs("results/stage2", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# n_jobs=-1 added to every model that supports it
# 3 slow SVMs (RBF, Poly, Sigmoid) removed
# ─────────────────────────────────────────────────────────────
def get_models():
    return [
        ("LogisticRegression",
            LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),

        ("SVM_Linear",
            SVC(kernel="linear", probability=True, random_state=42)),
            # SVM doesn't support n_jobs but linear kernel is fast enough

        ("kNN",
            KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),

        ("GaussianNB",
            GaussianNB()),
            # No n_jobs — already instant

        ("DecisionTree",
            DecisionTreeClassifier(random_state=42)),
            # No n_jobs — single tree

        ("RandomForest",
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),

        ("ExtraTrees",
            ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)),

        ("AdaBoost",
            AdaBoostClassifier(n_estimators=100, random_state=42)),
            # AdaBoost is sequential by design — no n_jobs

        ("GradientBoosting",
            GradientBoostingClassifier(n_estimators=100, random_state=42)),
            # sklearn GBM is sequential — no n_jobs

        ("XGBoost",
            XGBClassifier(
                n_estimators=100, random_state=42,
                eval_metric="logloss", verbosity=0,
                n_jobs=-1, tree_method="hist"
                # tree_method=hist is faster for large datasets
            )),

        ("LightGBM",
            LGBMClassifier(
                n_estimators=100, random_state=42,
                verbose=-1, n_jobs=-1
            )),

        ("CatBoost",
            CatBoostClassifier(
                iterations=100, random_state=42,
                verbose=0, thread_count=-1
                # CatBoost uses thread_count instead of n_jobs
            )),

        ("MLP",
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=300, random_state=42
                # No n_jobs — MLP is single-threaded in sklearn
            )),
    ]


# ─────────────────────────────────────────────────────────────
# SCREENING FUNCTION
# ─────────────────────────────────────────────────────────────
def screen_models(X, y, stage_name, out_dir, n_splits=5):
    print(f"\n{'='*60}")
    print(f"  SCREENING: {stage_name}")
    print(f"  Dataset : {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Classes : {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Folds   : {n_splits}-fold stratified CV + SMOTE")
    print(f"{'='*60}\n")

    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = get_models()
    all_results = []

    # Track progress
    total_models = len(models)

    for m_idx, (model_name, model) in enumerate(models, 1):
        print(f"  [{m_idx:02d}/{total_models}] {model_name}")
        model_start = time.time()
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # SMOTE only on training fold — never on validation
            smote = SMOTE(random_state=42)
            X_tr_bal, y_tr_bal = smote.fit_resample(X_tr, y_tr)

            # Pipeline: scale → classify
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    model)
            ])
            pipe.fit(X_tr_bal, y_tr_bal)

            y_pred = pipe.predict(X_val)
            y_prob = (
                pipe.predict_proba(X_val)[:, 1]
                if hasattr(model, "predict_proba")
                else y_pred.astype(float)
            )

            fold_metrics.append({
                "fold":             fold,
                "accuracy":         accuracy_score(y_val, y_pred),
                "f1_macro":         f1_score(y_val, y_pred, average="macro"),
                "f1_weighted":      f1_score(y_val, y_pred, average="weighted"),
                "balanced_accuracy":balanced_accuracy_score(y_val, y_pred),
                "mcc":              matthews_corrcoef(y_val, y_pred),
                "roc_auc":          roc_auc_score(y_val, y_prob),
                "precision":        precision_score(y_val, y_pred, zero_division=0),
                "recall":           recall_score(y_val, y_pred, zero_division=0),
            })

        # ── Aggregate across 5 folds ──────────────────────────
        fold_df = pd.DataFrame(fold_metrics)
        row = {"model": model_name}

        for col in ["accuracy","f1_macro","f1_weighted","balanced_accuracy",
                    "mcc","roc_auc","precision","recall"]:
            row[f"{col}_mean"] = round(fold_df[col].mean(), 4)
            row[f"{col}_std"]  = round(fold_df[col].std(),  4)

        # Ranking formula
        row["ranking_score"] = round(
            0.6 * row["f1_macro_mean"]
          + 0.3 * row["balanced_accuracy_mean"]
          - 0.1 * row["f1_macro_std"],
            4
        )

        all_results.append(row)

        elapsed = (time.time() - model_start) / 60
        print(f"         Acc={row['accuracy_mean']:.4f} "
              f"| F1={row['f1_macro_mean']:.4f}±{row['f1_macro_std']:.4f} "
              f"| AUC={row['roc_auc_mean']:.4f} "
              f"| MCC={row['mcc_mean']:.4f} "
              f"| Prec={row['precision_mean']:.4f} "
              f"| Rec={row['recall_mean']:.4f} "
              f"| Rank={row['ranking_score']:.4f} "
              f"| Time={elapsed:.1f}min")

        # ── Save results after EVERY model ───────────────────
        # So if script crashes, you don't lose completed models
        partial_df = pd.DataFrame(all_results).sort_values("ranking_score", ascending=False)
        partial_path = f"{out_dir}/screening_results.csv"
        partial_df.to_csv(partial_path, index=False)
        print(f"         💾 Saved so far → {partial_path}")

    # ── Final sorted results ──────────────────────────────────
    results_df = pd.DataFrame(all_results).sort_values("ranking_score", ascending=False)
    results_df.to_csv(f"{out_dir}/screening_results.csv", index=False)

    # ── Top 5 ─────────────────────────────────────────────────
    top5 = results_df.head(5)
    top5.to_csv(f"{out_dir}/top5_models.csv", index=False)

    # ── Plot: model comparison bar chart ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{stage_name} — Model Screening Results", fontsize=14, fontweight="bold")

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_df)))

    for ax, metric, title in zip(
        axes,
        ["f1_macro_mean", "roc_auc_mean", "mcc_mean"],
        ["F1-Macro", "ROC-AUC", "MCC"]
    ):
        bars = ax.barh(
            results_df["model"],
            results_df[metric],
            xerr=results_df.get(metric.replace("mean","std"), None),
            color=colors, alpha=0.85, edgecolor="white"
        )
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.set_xlim(results_df[metric].min() - 0.05, 1.0)
        ax.axvline(results_df[metric].mean(), color="red",
                   linestyle="--", linewidth=1, label="Mean")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = f"{out_dir}/screening_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Plot saved → {plot_path}")

    # ── Print final ranking ───────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  FINAL RANKING — {stage_name}")
    print(f"  {'─'*60}")
    print(f"  {'Rank':<5} {'Model':<22} {'F1-Macro':>10} {'AUC':>8} {'MCC':>8} {'Score':>8}")
    print(f"  {'─'*60}")
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"  {i:<5} {row['model']:<22} "
              f"{row['f1_macro_mean']:>8.4f}  "
              f"{row['roc_auc_mean']:>8.4f}  "
              f"{row['mcc_mean']:>8.4f}  "
              f"{row['ranking_score']:>8.4f}")

    print(f"\n  🏆 Top 5 saved → {out_dir}/top5_models.csv")

    return results_df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    total_start = time.time()

    # ── STAGE 1 ───────────────────────────────────────────────
    print("\nLoading Stage 1 features...")
    X1 = np.load("data/processed/stage1_features.npy")
    y1 = np.load("data/processed/stage1_labels.npy")

    screen_models(X1, y1, "Stage 1 — AI vs Human", "results/stage1")

    # ── STAGE 2 ───────────────────────────────────────────────
    print("\nLoading Stage 2 features...")
    X2 = np.load("data/processed/stage2_features.npy")
    y2 = np.load("data/processed/stage2_labels.npy")

    screen_models(X2, y2, "Stage 2 — Unsafe", "results/stage2")

    total_mins = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"  ALL SCREENING DONE in {total_mins:.1f} minutes")
    print(f"{'='*60}")
    print(f"\n  Results:")
    print(f"    results/stage1/screening_results.csv")
    print(f"    results/stage1/top5_models.csv")
    print(f"    results/stage2/screening_results.csv")
    print(f"    results/stage2/top5_models.csv")