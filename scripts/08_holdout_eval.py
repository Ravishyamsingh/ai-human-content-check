"""
Final holdout evaluation — run ONCE at the very end.
Never run this during development.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, matthews_corrcoef)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_on_holdout(model_path, holdout_X_path, holdout_y_path, stage_name, out_dir):
    print(f"\n{'='*60}")
    print(f"FINAL HOLDOUT EVALUATION: {stage_name}")
    print(f"{'='*60}")

    # Load
    model = joblib.load(model_path)
    X_holdout = np.load(holdout_X_path)
    y_holdout = np.load(holdout_y_path)

    # Predict
    y_pred = model.predict(X_holdout)
    y_prob = model.predict_proba(X_holdout)[:, 1]

    # Report
    report = classification_report(y_holdout, y_pred)
    mcc = matthews_corrcoef(y_holdout, y_pred)
    auc = roc_auc_score(y_holdout, y_prob)

    print(report)
    print(f"MCC: {mcc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    # Save text report
    with open(f"{out_dir}/holdout_report.txt", "w") as f:
        f.write(f"FINAL HOLDOUT EVALUATION: {stage_name}\n")
        f.write("="*60 + "\n")
        f.write(report)
        f.write(f"\nMCC: {mcc:.4f}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_holdout, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix — {stage_name}")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/holdout_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"✅ Holdout report saved to {out_dir}/")


if __name__ == "__main__":
    evaluate_on_holdout(
        "models/stage1_authorship.pkl",
        "data/holdout/stage1_holdout_X.npy",
        "data/holdout/stage1_holdout_y.npy",
        "Stage 1 - AI vs Human",
        "results/stage1"
    )
    evaluate_on_holdout(
        "models/stage2_jailbreak.pkl",
        "data/holdout/stage2_holdout_X.npy",
        "data/holdout/stage2_holdout_y.npy",
        "Stage 2 - Jailbreak",
        "results/stage2"
    )