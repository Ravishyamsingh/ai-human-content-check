"""
Download both datasets from HuggingFace.
Saves raw CSVs to data/raw/
"""
import os
import pandas as pd
from datasets import load_dataset

os.makedirs("data/raw", exist_ok=True)

# ─────────────────────────────────────────────
# STAGE 1 — AI vs Human
# ─────────────────────────────────────────────
print("⬇️  Downloading Stage 1 dataset...")
ds1 = load_dataset("silentone0725/ai-human-text-detection-v1")

# Convert to dataframe
df1 = ds1["train"].to_pandas()
print(f"Stage 1 shape: {df1.shape}")
print(f"Stage 1 columns: {df1.columns.tolist()}")
print(f"Stage 1 class distribution:\n{df1['label'].value_counts()}")

# Save
df1.to_csv("data/raw/stage1_raw.csv", index=False)
print("✅ Stage 1 saved → data/raw/stage1_raw.csv")

# ─────────────────────────────────────────────
# STAGE 2 — Jailbreak Detection
# ─────────────────────────────────────────────
print("\n⬇️  Downloading Stage 2 dataset...")
ds2 = load_dataset("BallAdMyFi/jailbreaking_prompt_v2")

df2 = ds2["train"].to_pandas()
print(f"Stage 2 shape: {df2.shape}")
print(f"Stage 2 columns: {df2.columns.tolist()}")
print(df2.head(2))

# Save
df2.to_csv("data/raw/stage2_raw.csv", index=False)
print("✅ Stage 2 saved → data/raw/stage2_raw.csv")