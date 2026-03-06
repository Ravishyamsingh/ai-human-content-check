"""
╔══════════════════════════════════════════════════════════════╗
║  STAGE: PREPROCESSING                                        ║
║  Philosophy: Minimal intervention, maximum signal retention  ║
║                                                              ║
║  We ONLY fix data quality issues.                            ║
║  We NEVER touch writing style.                               ║
║                                                              ║
║  Justification: This is a stylometric system. Capitalization,║
║  punctuation, sentence structure ARE the features.           ║
║  Removing them defeats the purpose of the entire system.     ║
╚══════════════════════════════════════════════════════════════╝

What this script does (and WHY — thesis justification included):

  STEP 1 — Select columns
           Keep only text + label. Drop everything else.

  STEP 2 — Convert to string
           Safety step. Some rows may be float/int due to CSV parsing.

  STEP 3 — Fix invisible encoding artifacts
           HTML entities (&amp; &lt; &gt;) → real characters
           Zero-width / invisible unicode → removed
           These are NOT stylistic — they are encoding noise.

  STEP 4 — Normalize whitespace ONLY
           Multiple spaces → single space
           Tabs/carriage returns → single space
           Leading/trailing whitespace → stripped
           We do NOT remove newlines within text
           (paragraph breaks are structural signals)

  STEP 5 — Remove nulls
           Cannot process NaN rows.

  STEP 6 — Remove exact duplicates (by text content)
           Exact duplicate samples would bias model training.
           We use MD5 hash for speed.

  STEP 7 — Remove texts under 10 words
           Texts with fewer than 10 words cannot carry
           meaningful stylometric or semantic signal.
           Threshold of 10 words is conservative and defensible.
           (Stamatatos 2009: minimum viable stylometric unit)

  STEP 8 — Ensure label is integer (0 or 1)
           Standardize label dtype across both datasets.

WHAT WE DO NOT DO (and why, for your thesis):
  ✗ Lowercase    → destroys capitalization signal
  ✗ Remove punct → destroys punctuation density signal
  ✗ Stemming     → destroys vocabulary richness signal
  ✗ Stopword rm  → stopword ratio IS a feature we extract
  ✗ Aggressive length cutoff → would bias against long-form text
"""

import os
import re
import html
import hashlib
import unicodedata
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

os.makedirs("data/processed", exist_ok=True)
os.makedirs("results/preprocessing", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# AUDIT LOG
# Every number here goes into your thesis Table 3.x
# ─────────────────────────────────────────────────────────────
audit_lines = []

def log(msg):
    print(msg)
    audit_lines.append(msg)

def save_audit():
    path = "results/preprocessing/preprocessing_audit.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("PREPROCESSING AUDIT REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        for line in audit_lines:
            f.write(line + "\n")
    print(f"\n✅ Audit log → {path}")


# ─────────────────────────────────────────────────────────────
# SINGLE TEXT CLEANER
# Only touches encoding artifacts. Nothing stylistic.
# ─────────────────────────────────────────────────────────────
def fix_encoding_artifacts(text):
    """
    Fix encoding artifacts that are NOT part of the original writing.

    These are technical artifacts introduced by web scraping,
    dataset collection pipelines, or CSV encoding — not the author's style.

    Applied operations (in order):
      1. HTML entity decoding  — &amp; → &,  &lt; → <,  &quot; → "
      2. Unicode normalization (NFC) — compose decomposed characters
      3. Remove zero-width / invisible unicode chars
         (zero-width space, zero-width non-joiner, BOM, soft hyphen)
      4. Normalize internal whitespace
         (tabs, carriage returns → space; multiple spaces → one space)
      5. Strip leading/trailing whitespace
    """
    # Step 1: HTML entity decode
    text = html.unescape(text)

    # Step 2: Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Step 3: Remove invisible/zero-width unicode characters
    # These are NOT visible, NOT stylistic — pure encoding noise
    invisible_chars = [
        '\u200b',  # zero-width space
        '\u200c',  # zero-width non-joiner
        '\u200d',  # zero-width joiner
        '\ufeff',  # byte order mark (BOM)
        '\u00ad',  # soft hyphen
        '\u2060',  # word joiner
        '\u180e',  # mongolian vowel separator
    ]
    for ch in invisible_chars:
        text = text.replace(ch, '')

    # Step 4: Normalize whitespace
    # Replace tabs and carriage returns with space
    text = re.sub(r'[\t\r]', ' ', text)
    # Collapse multiple spaces into one
    text = re.sub(r' {2,}', ' ', text)

    # Step 5: Strip
    text = text.strip()

    return text


def get_md5(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────
# MASTER CLEANING FUNCTION
# ─────────────────────────────────────────────────────────────
def clean_dataset(df_raw, text_col, label_col, stage_name, min_words=10):
    """
    Applies the minimal-intervention preprocessing pipeline.

    Parameters
    ----------
    df_raw     : raw DataFrame as downloaded
    text_col   : column name containing text
    label_col  : column name containing label (0/1)
    stage_name : used for logging only
    min_words  : minimum word count to retain a sample (default: 10)
    """

    log(f"\n{'═'*70}")
    log(f"  PREPROCESSING: {stage_name}")
    log(f"{'═'*70}")
    log(f"\n  Raw shape          : {df_raw.shape}")
    log(f"  Text column        : '{text_col}'")
    log(f"  Label column       : '{label_col}'")
    log(f"\n  RAW class distribution:")
    vc = df_raw[label_col].value_counts().sort_index()
    for cls, cnt in vc.items():
        log(f"    Class {cls}: {cnt:,} samples")
    raw_ratio = vc.max() / vc.min()
    log(f"  Imbalance ratio    : {raw_ratio:.2f}:1")

    log(f"\n  --- Cleaning Steps ---")

    # ── STEP 1: Select columns ────────────────────────────────
    df = df_raw[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    n = len(df)
    log(f"\n  Step 1 | Select text + label columns")
    log(f"    Rows retained: {n:,}")

    # ── STEP 2: Convert to string ─────────────────────────────
    df["text"] = df["text"].astype(str)
    log(f"\n  Step 2 | Convert text column to string dtype")
    log(f"    Applied to all {n:,} rows")

    # ── STEP 3: Fix encoding artifacts ───────────────────────
    df["text"] = df["text"].apply(fix_encoding_artifacts)
    log(f"\n  Step 3 | Fix encoding artifacts")
    log(f"    HTML entities decoded, invisible unicode removed,")
    log(f"    whitespace normalized. No stylistic changes.")

    # ── STEP 4: Remove nulls ──────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    removed = before - len(df)
    log(f"\n  Step 4 | Remove null rows")
    log(f"    Before: {before:,}  |  After: {len(df):,}  |  Removed: {removed:,}")

    # ── STEP 5: Remove exact duplicates ──────────────────────
    before = len(df)
    df["_hash"] = df["text"].apply(get_md5)
    df = df.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"])
    removed = before - len(df)
    pct = removed / before * 100 if before > 0 else 0
    log(f"\n  Step 5 | Remove exact duplicates (MD5 hash comparison)")
    log(f"    Before: {before:,}  |  After: {len(df):,}  |  Removed: {removed:,} ({pct:.2f}%)")

    # ── STEP 6: Remove texts under minimum word count ─────────
    before = len(df)
    df["_wc"] = df["text"].apply(lambda x: len(x.split()))
    short_count = (df["_wc"] < min_words).sum()
    df = df[df["_wc"] >= min_words].drop(columns=["_wc"])
    removed = before - len(df)
    pct = removed / before * 100 if before > 0 else 0
    log(f"\n  Step 6 | Remove texts with fewer than {min_words} words")
    log(f"    Rationale: < {min_words} words = insufficient stylometric signal")
    log(f"    Before: {before:,}  |  After: {len(df):,}  |  Removed: {removed:,} ({pct:.2f}%)")


    # ── STEP 7: Ensure label is integer ──────────────────────
    # Handles both numeric (0/1) and string ("human"/"ai", "safe"/"unsafe")
    if df["label"].dtype == object:
        sorted_labels = sorted(df["label"].astype(str).unique())
        label_map = {l: i for i, l in enumerate(sorted_labels)}
        df["label"] = df["label"].astype(str).map(label_map)
        log(f"  Step 7 [Label encoding]: String→int mapping applied: {label_map}")
    else:
        df["label"] = df["label"].astype(int)
        log(f"  Step 7 [Label encoding]: Converted to int")
    log(f"    Final label values: {sorted(df['label'].unique())}")

    # ── STEP 8: Reset index ───────────────────────────────────
    df = df.reset_index(drop=True)

    # ── FINAL SUMMARY ─────────────────────────────────────────
    log(f"\n  {'─'*60}")
    log(f"  FINAL SUMMARY: {stage_name}")
    log(f"  {'─'*60}")
    log(f"  Final shape: {df.shape}")
    log(f"\n  CLEAN class distribution:")
    vc2 = df["label"].value_counts().sort_index()
    for cls, cnt in vc2.items():
        log(f"    Class {cls}: {cnt:,} samples")
    if vc2.min() > 0:
        final_ratio = vc2.max() / vc2.min()
        log(f"  Imbalance ratio (clean): {final_ratio:.2f}:1")

    wc = df["text"].apply(lambda x: len(x.split()))
    log(f"\n  Word count stats (clean dataset):")
    log(f"    Mean   : {wc.mean():.1f}")
    log(f"    Median : {wc.median():.1f}")
    log(f"    Std    : {wc.std():.1f}")
    log(f"    Min    : {wc.min()}")
    log(f"    Max    : {wc.max()}")

    overall_removed = len(df_raw) - len(df)
    overall_pct = overall_removed / len(df_raw) * 100 if len(df_raw) > 0 else 0
    log(f"\n  Total removed: {overall_removed:,} samples ({overall_pct:.2f}% of raw)")
    log(f"  Total retained: {len(df):,} samples ({100-overall_pct:.2f}% of raw)")

    return df


# ─────────────────────────────────────────────────────────────
# CROSS-DATASET CONTAMINATION CHECK
# Are any texts from Stage 1 present in Stage 2?
# Examiners WILL ask this. Answer must be documented.
# ─────────────────────────────────────────────────────────────
def check_cross_contamination(df1, df2):
    log(f"\n{'═'*70}")
    log(f"  CROSS-DATASET CONTAMINATION CHECK")
    log(f"{'═'*70}")
    log(f"  Checking if any Stage 1 texts appear in Stage 2...")

    hashes1 = set(df1["text"].apply(get_md5))
    hashes2 = set(df2["text"].apply(get_md5))
    overlap = hashes1 & hashes2

    log(f"  Stage 1 unique texts : {len(hashes1):,}")
    log(f"  Stage 2 unique texts : {len(hashes2):,}")
    log(f"  Overlapping texts    : {len(overlap)}")

    if len(overlap) == 0:
        log(f"  ✅ RESULT: No cross-contamination detected. Datasets are clean.")
    else:
        log(f"  ⚠️  WARNING: {len(overlap)} texts appear in both datasets!")
        log(f"  These should be investigated before proceeding.")

    return len(overlap)


# ─────────────────────────────────────────────────────────────
# PREPROCESSING VISUALIZATIONS
# Side-by-side before vs after plots
# ─────────────────────────────────────────────────────────────
def plot_before_after(df_raw, df_clean, label_col_raw, stage_name, label_names, out_dir):
    """
    Generates before vs after comparison plots for the thesis.
    Saves to results/preprocessing/
    """
    os.makedirs(out_dir, exist_ok=True)
    prefix = stage_name.lower().replace(" ", "_").replace("—", "").replace("-", "_")

    # ── Plot 1: Class distribution before vs after ────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{stage_name} — Class Distribution: Before vs After Preprocessing",
                 fontsize=13, fontweight="bold")

    # Before
    vc_before = df_raw[label_col_raw].value_counts().sort_index()
    colors = ["#4C72B0", "#DD8452"]
    axes[0].bar([str(k) for k in vc_before.index], vc_before.values, color=colors[:len(vc_before)])
    axes[0].set_title("Before Preprocessing", fontsize=11)
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Sample Count")
    for i, v in enumerate(vc_before.values):
        axes[0].text(i, v + vc_before.max() * 0.01, f"{v:,}", ha="center", fontsize=10)

    # After
    vc_after = df_clean["label"].value_counts().sort_index()
    axes[1].bar([label_names.get(k, str(k)) for k in vc_after.index],
                vc_after.values, color=colors[:len(vc_after)])
    axes[1].set_title("After Preprocessing", fontsize=11)
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Sample Count")
    for i, v in enumerate(vc_after.values):
        axes[1].text(i, v + vc_after.max() * 0.01, f"{v:,}", ha="center", fontsize=10)

    plt.tight_layout()
    fname = f"{out_dir}/{prefix}_class_distribution_before_after.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {fname}")

    # ── Plot 2: Word count distribution before vs after ───────
    df_raw_copy = df_raw.copy()
    text_col = [c for c in df_raw.columns if "text" in c.lower()][0]
    df_raw_copy["wc"] = df_raw_copy[text_col].astype(str).apply(lambda x: len(x.split()))
    df_clean_copy = df_clean.copy()
    df_clean_copy["wc"] = df_clean_copy["text"].apply(lambda x: len(x.split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"{stage_name} — Word Count Distribution: Before vs After",
                 fontsize=13, fontweight="bold")

    axes[0].hist(df_raw_copy["wc"].clip(0, 1000), bins=60, color="#4C72B0", alpha=0.8, edgecolor="white")
    axes[0].set_title("Before Preprocessing")
    axes[0].set_xlabel("Word Count (clipped at 1000)")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(df_raw_copy["wc"].median(), color="red", linestyle="--",
                    label=f"Median={df_raw_copy['wc'].median():.0f}")
    axes[0].legend()

    axes[1].hist(df_clean_copy["wc"].clip(0, 1000), bins=60, color="#DD8452", alpha=0.8, edgecolor="white")
    axes[1].set_title("After Preprocessing")
    axes[1].set_xlabel("Word Count (clipped at 1000)")
    axes[1].set_ylabel("Frequency")
    axes[1].axvline(df_clean_copy["wc"].median(), color="red", linestyle="--",
                    label=f"Median={df_clean_copy['wc'].median():.0f}")
    axes[1].legend()

    plt.tight_layout()
    fname = f"{out_dir}/{prefix}_word_count_before_after.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved: {fname}")


# ═════════════════════════════════════════════════════════════
# MAIN — Run both stages
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":

    log(f"PREPROCESSING PIPELINE START")
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Load raw data ─────────────────────────────────────────
    df1_raw = pd.read_csv("data/raw/stage1_raw.csv")
    df2_raw = pd.read_csv("data/raw/stage2_raw.csv")

    log(f"\nStage 1 columns: {df1_raw.columns.tolist()}")
    log(f"Stage 2 columns: {df2_raw.columns.tolist()}")

    # ⚠️  CHECK THESE AFTER RUNNING 01_download_data.py
    # Open the raw CSVs and look at what columns actually exist.
    # Then update the values below.
    TEXT_COL_S1  = "text"    # ← update if different
    LABEL_COL_S1 = "label"   # ← update if different
    TEXT_COL_S2  = "text"    # ← update if different
    LABEL_COL_S2 = "label"   # ← update if different

    # ── Stage 1 ───────────────────────────────────────────────
    df1_clean = clean_dataset(
        df1_raw,
        text_col   = TEXT_COL_S1,
        label_col  = LABEL_COL_S1,
        stage_name = "Stage 1 — AI vs Human",
        min_words  = 10
    )
    df1_clean.to_csv("data/processed/stage1_clean.csv", index=False)
    log(f"\n  ✅ Stage 1 clean CSV → data/processed/stage1_clean.csv")

    plot_before_after(
        df1_raw, df1_clean,
        label_col_raw = LABEL_COL_S1,
        stage_name    = "Stage 1 — AI vs Human",
        label_names   = {0: "Human", 1: "AI"},
        out_dir       = "results/preprocessing"
    )

    # ── Stage 2 ───────────────────────────────────────────────
    df2_clean = clean_dataset(
        df2_raw,
        text_col   = TEXT_COL_S2,
        label_col  = LABEL_COL_S2,
        stage_name = "Stage 2 — Unsafe Prompt Detection",
        min_words  = 10
    )
    df2_clean.to_csv("data/processed/stage2_clean.csv", index=False)
    log(f"\n  ✅ Stage 2 clean CSV → data/processed/stage2_clean.csv")

    plot_before_after(
        df2_raw, df2_clean,
        label_col_raw = LABEL_COL_S2,
        stage_name    = "Stage 2 — Unsafe Prompt Detection",
        label_names   = {0: "Safe", 1: "Unsafe"},
        out_dir       = "results/preprocessing"
    )

    # ── Cross-contamination check ─────────────────────────────
    check_cross_contamination(df1_clean, df2_clean)

    # ── Save audit log ────────────────────────────────────────
    save_audit()

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Stage 1: {len(df1_clean):,} samples → data/processed/stage1_clean.csv")
    print(f"Stage 2: {len(df2_clean):,} samples → data/processed/stage2_clean.csv")
    print(f"Audit   : results/preprocessing/preprocessing_audit.txt")
    print(f"Plots   : results/preprocessing/")