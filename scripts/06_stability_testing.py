"""
╔══════════════════════════════════════════════════════════════╗
║  STAGE: STABILITY TESTING                                    ║
║                                                              ║
║  PURPOSE (for your thesis viva defense):                     ║
║                                                              ║
║  When you report "my model got 94% F1" — the examiner will  ║
║  immediately ask: "How do you know that's not just luck      ║
║  from one particular train/test split?"                      ║
║                                                              ║
║  Stability testing answers this by:                          ║
║    1. Training the SAME model 5 times                        ║
║    2. Each time with a DIFFERENT random seed                 ║
║       (different train/test split each time)                 ║
║    3. Measuring how CONSISTENT the results are               ║
║                                                              ║
║  If mean=0.932 and std=0.002 → STABLE → trustworthy         ║
║  If mean=0.903 and std=0.038 → UNSTABLE → questionable       ║
║                                                              ║
║  Your thesis answer to the examiner:                         ║
║  "Stability testing across 5 random seeds showed a          ║
║   standard deviation of ±0.002 in F1-Macro, confirming      ║
║   that results are not seed-dependent artifacts."            ║
║                                                              ║
║  What this script does:                                      ║
║    - Takes Top 5 models from screening results               ║
║    - Runs each model 5 times with seeds [42,123,456,789,2024]║
║    - Records F1, AUC, Balanced Accuracy each time            ║
║    - Computes mean ± std for each model                      ║
║    - Plots stability comparison chart                        ║
║    - Saves full results CSV                                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE

# All models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

os.makedirs("results/stage1", exist_ok=True)
os.makedirs("results/stage2", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# SEEDS — the 5 different random scenarios we test
#
# Each seed gives a completely different train/test split.
# If model performance is consistent across all 5 → stable.
# If it varies wildly → unstable → not trustworthy.
# ─────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 2024]


# ─────────────────────────────────────────────────────────────
# MODEL MAP — same as screening, used to reconstruct models
# ─────────────────────────────────────────────────────────────
def get_model(name):
    """Return a fresh model instance by name."""
    model_map = {
        "LogisticRegression":  LogisticRegression(max_iter=1000, random_state=42),
        "SVM_Linear":          SVC(kernel="linear", probability=True, random_state=42),
        "kNN":                 KNeighborsClassifier(n_neighbors=5),
        "GaussianNB":          GaussianNB(),
        "DecisionTree":        DecisionTreeClassifier(random_state=42),
        "RandomForest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "ExtraTrees":          ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
        "GradientBoosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                             eval_metric="logloss", verbosity=0),
        "LightGBM":            LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        "CatBoost":            CatBoostClassifier(iterations=100, random_state=42, verbose=0),
        "MLP":                 MLPClassifier(hidden_layer_sizes=(256,128),
                                             max_iter=300, random_state=42),
    }
    return model_map.get(name)


# ─────────────────────────────────────────────────────────────
# SINGLE MODEL STABILITY TEST
#
# For ONE model, runs training+evaluation 5 times
# (once per seed = once per different random split).
#
# Returns a dict of results including mean, std, min, max
# for F1-Macro, ROC-AUC, Balanced Accuracy.
# ─────────────────────────────────────────────────────────────
def test_model_stability(model_name, X, y, seeds=SEEDS, test_size=0.20):
    """
    Parameters
    ----------
    model_name : string name of the model
    X          : feature matrix (n_samples, 420)
    y          : labels (n_samples,)
    seeds      : list of random seeds to test
    test_size  : proportion held out for evaluation each run

    Returns
    -------
    dict with aggregated results across all seeds
    """
    f1_scores  = []
    auc_scores = []
    bal_scores = []

    per_seed_results = []

    for seed in seeds:
        # Different split for each seed
        # This is what "stability" means — we test on different data each time
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size    = test_size,
            random_state = seed,
            stratify     = y       # preserve class proportions
        )

        # Apply SMOTE only on training data (NEVER on test)
        smote = SMOTE(random_state=seed)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

        # Build pipeline
        model = get_model(model_name)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    model)
        ])

        # Train
        pipe.fit(X_train_bal, y_train_bal)

        # Predict
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        # Metrics
        f1  = f1_score(y_test, y_pred, average="macro")
        auc = roc_auc_score(y_test, y_prob)
        bal = balanced_accuracy_score(y_test, y_pred)

        f1_scores.append(f1)
        auc_scores.append(auc)
        bal_scores.append(bal)

        per_seed_results.append({
            "seed": seed,
            "f1_macro": f1,
            "roc_auc": auc,
            "balanced_accuracy": bal
        })

    # Aggregate
    result = {
        "model": model_name,

        # F1-Macro across 5 seeds
        "f1_mean":  np.mean(f1_scores),
        "f1_std":   np.std(f1_scores),
        "f1_min":   np.min(f1_scores),
        "f1_max":   np.max(f1_scores),
        "f1_range": np.max(f1_scores) - np.min(f1_scores),  # how much it varies

        # ROC-AUC across 5 seeds
        "auc_mean": np.mean(auc_scores),
        "auc_std":  np.std(auc_scores),

        # Balanced Accuracy across 5 seeds
        "bal_mean": np.mean(bal_scores),
        "bal_std":  np.std(bal_scores),

        # Stability score: 1 - std (higher = more stable)
        # A model with std=0.001 gets stability=0.999
        # A model with std=0.05  gets stability=0.950
        "stability_score": round(1 - np.std(f1_scores), 6),

        # Per-seed detail
        "per_seed": per_seed_results,
        "f1_values": f1_scores,  # kept for plotting
    }

    return result


# ─────────────────────────────────────────────────────────────
# STABILITY VISUALIZATION
# ─────────────────────────────────────────────────────────────
def plot_stability_results(all_results, stage_name, out_path):
    """
    Plot 1: Mean F1 ± std error bars for each model
    Plot 2: Per-seed F1 line chart (shows how each model varies)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{stage_name} — Stability Analysis Across 5 Random Seeds",
                 fontsize=13, fontweight="bold")

    model_names = [r["model"] for r in all_results]
    means       = [r["f1_mean"] for r in all_results]
    stds        = [r["f1_std"]  for r in all_results]

    # ── Left: Mean ± Std bar chart ────────────────────────────
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    bars = axes[0].bar(range(len(model_names)), means, yerr=stds,
                       color=colors, capsize=6, alpha=0.85,
                       edgecolor="white", error_kw={"linewidth": 2})
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, rotation=35, ha="right", fontsize=9)
    axes[0].set_ylabel("F1-Macro (Mean ± Std)")
    axes[0].set_title("Mean F1 with Stability Error Bars\n(Smaller bar = more stable)")
    axes[0].set_ylim(max(0, min(means) - 0.05), min(1.0, max(means) + 0.05))

    # Annotate bars
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, m + s + 0.002, f"μ={m:.3f}\nσ={s:.4f}",
                     ha="center", fontsize=7.5)

    # ── Right: Per-seed line chart ────────────────────────────
    # Shows EXACTLY how each model performed on each seed
    for i, result in enumerate(all_results):
        f1_vals = result["f1_values"]
        axes[1].plot(SEEDS, f1_vals, marker="o", label=result["model"],
                     color=colors[i], linewidth=1.5, markersize=5)

    axes[1].set_xlabel("Random Seed")
    axes[1].set_ylabel("F1-Macro")
    axes[1].set_title("Per-Seed F1 Trajectories\n(Flat line = very stable)")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_xticks(SEEDS)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Plot saved → {out_path}")


def plot_stability_heatmap(all_results, stage_name, out_path):
    """
    Heatmap: models × seeds — shows the F1 value for each seed
    Good visual for thesis to show consistency
    """
    n_models = len(all_results)
    n_seeds  = len(SEEDS)

    data = np.zeros((n_models, n_seeds))
    for i, result in enumerate(all_results):
        data[i, :] = result["f1_values"]

    model_names = [r["model"] for r in all_results]

    fig, ax = plt.subplots(figsize=(10, max(4, n_models * 0.7)))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=[f"seed={s}" for s in SEEDS],
                yticklabels=model_names,
                ax=ax, linewidths=0.5, vmin=0.7, vmax=1.0,
                annot_kws={"size": 10})
    ax.set_title(f"{stage_name} — F1-Macro per Seed (Stability Heatmap)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Model")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Heatmap saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# MASTER STABILITY RUNNER
# ─────────────────────────────────────────────────────────────
def run_stability_testing(X, y, screening_csv, stage_name, out_dir):
    """
    Reads top 5 models from screening results.
    Runs stability test for each.
    Saves results + plots.
    """
    print(f"\n{'═'*70}")
    print(f"  STABILITY TESTING: {stage_name}")
    print(f"{'═'*70}")
    print(f"  Logic: Each model is trained+evaluated {len(SEEDS)} times,")
    print(f"         once per random seed: {SEEDS}")
    print(f"         Each seed = different train/test split.")
    print(f"         Low std across seeds = stable = trustworthy result.")
    print()

    # Get top 5 models
    screening_df = pd.read_csv(screening_csv)
    top5_names   = screening_df.head(5)["model"].tolist()
    print(f"  Top 5 models to test: {top5_names}\n")

    all_results  = []
    per_seed_rows = []

    for rank, model_name in enumerate(top5_names, 1):
        print(f"  [{rank}/5] Testing: {model_name}")
        result = test_model_stability(model_name, X, y)

        print(f"    F1-Macro : {result['f1_mean']:.4f} ± {result['f1_std']:.4f}")
        print(f"    ROC-AUC  : {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")
        print(f"    Stability: {result['stability_score']:.4f}  "
              f"(range={result['f1_range']:.4f})")
        print(f"    Per-seed F1: {[round(v,4) for v in result['f1_values']]}")
        print()

        all_results.append(result)

        # Collect per-seed detail rows
        for ps in result["per_seed"]:
            per_seed_rows.append({
                "model":             model_name,
                "seed":              ps["seed"],
                "f1_macro":          ps["f1_macro"],
                "roc_auc":           ps["roc_auc"],
                "balanced_accuracy": ps["balanced_accuracy"],
            })

    # ── Summary CSV ───────────────────────────────────────────
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "model":           r["model"],
            "f1_mean":         r["f1_mean"],
            "f1_std":          r["f1_std"],
            "f1_min":          r["f1_min"],
            "f1_max":          r["f1_max"],
            "f1_range":        r["f1_range"],
            "auc_mean":        r["auc_mean"],
            "auc_std":         r["auc_std"],
            "bal_mean":        r["bal_mean"],
            "bal_std":         r["bal_std"],
            "stability_score": r["stability_score"],
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("f1_mean", ascending=False)
    summary_df.to_csv(f"{out_dir}/stability_results.csv", index=False)
    print(f"  ✅ Summary → {out_dir}/stability_results.csv")

    # Per-seed detail CSV
    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(f"{out_dir}/stability_per_seed.csv", index=False)
    print(f"  ✅ Per-seed detail → {out_dir}/stability_per_seed.csv")

    # ── Plots ─────────────────────────────────────────────────
    plot_stability_results(
        all_results, stage_name,
        f"{out_dir}/stability_comparison.png"
    )
    plot_stability_heatmap(
        all_results, stage_name,
        f"{out_dir}/stability_heatmap.png"
    )

    # ── Print final ranking ───────────────────────────────────
    print(f"\n  FINAL STABILITY RANKING — {stage_name}")
    print(f"  {'Model':<22} {'F1 Mean':>9} {'F1 Std':>9} {'Stability':>10}")
    print(f"  {'-'*55}")
    for _, row in summary_df.iterrows():
        print(f"  {row['model']:<22} {row['f1_mean']:>9.4f} "
              f"{row['f1_std']:>9.4f} {row['stability_score']:>10.4f}")

    best_stable = summary_df.iloc[0]["model"]
    print(f"\n  ✅ Most stable high-performing model: {best_stable}")
    print(f"     This model will be used for final training.")

    return summary_df


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── STAGE 1 ───────────────────────────────────────────────
    X1 = np.load("data/processed/stage1_features.npy")
    y1 = np.load("data/processed/stage1_labels.npy")
    run_stability_testing(
        X1, y1,
        screening_csv = "results/stage1/screening_results.csv",
        stage_name    = "Stage 1 — AI vs Human",
        out_dir       = "results/stage1"
    )

    # ── STAGE 2 ───────────────────────────────────────────────
    X2 = np.load("data/processed/stage2_features.npy")
    y2 = np.load("data/processed/stage2_labels.npy")
    run_stability_testing(
        X2, y2,
        screening_csv = "results/stage2/screening_results.csv",
        stage_name    = "Stage 2 — Unsafe Prompt Detection",
        out_dir       = "results/stage2"
    )

    print("\n" + "="*70)
    print("STABILITY TESTING COMPLETE")
    print("="*70)
    print("Results → results/stage1/stability_results.csv")
    print("          results/stage2/stability_results.csv")
    print("Plots   → results/stage1/stability_comparison.png")
    print("          results/stage1/stability_heatmap.png")
    print("          results/stage2/stability_comparison.png")
    print("          results/stage2/stability_heatmap.png")  