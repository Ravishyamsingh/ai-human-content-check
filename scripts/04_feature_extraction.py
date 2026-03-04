"""
FEATURE EXTRACTION — 420 DIMENSIONS
====================================
GPU-enabled + Per-block checkpoint saving.

Every single block is saved to disk the moment it finishes.
If script crashes, restart and it skips already-done blocks.

Checkpoint files:
    data/features/stage1_block_A_semantic.npy      (384)
    data/features/stage1_block_B_stylometric.npy   (10)
    data/features/stage1_block_C_statistical.npy   (7)
    data/features/stage1_block_D_linguistic.npy    (10)
    data/features/stage1_block_E_perplexity.npy    (1)
    data/features/stage1_block_F_structural.npy    (8)
    data/features/stage1_features.npy              (420 — final)
    data/features/stage1_labels.npy
    (same for stage2_*)

Also saves to:
    data/processed/stage1_features.npy
    data/processed/stage1_labels.npy
    data/processed/stage2_features.npy
    data/processed/stage2_labels.npy
so downstream scripts find them.
"""

import os
import re
import math
import time
import string
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
import spacy
import nltk

# Ensure NLTK stopwords are downloaded before importing
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# ─────────────────────────────────────────────────────────────
# DEVICE — uses GPU if available, falls back to CPU
# ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*60}")
print(f"  DEVICE: {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU   : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────
# DIRECTORIES
# ─────────────────────────────────────────────────────────────
os.makedirs("data/features",   exist_ok=True)
os.makedirs("data/processed",  exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STOPWORDS — from nltk (installed in setup step)
# ─────────────────────────────────────────────────────────────
stop_words = set(stopwords.words("english"))

# ─────────────────────────────────────────────────────────────
# LOAD ALL MODELS ONCE AT TOP
# ─────────────────────────────────────────────────────────────
print("Loading models...")

print("  [1/3] Loading sentence-transformers (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

print("  [2/3] Loading SpaCy...")
nlp = spacy.load("en_core_web_sm", disable=["ner"])

print("  [3/3] Loading GPT-2...")
gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt_model     = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_model.eval()
gpt_model.to(DEVICE)

print(f"  All models loaded on {DEVICE}\n")


# ═════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ═════════════════════════════════════════════════════════════

def ckpt_path(stage, block_name):
    """e.g. stage='stage1', block_name='A_semantic' → data/features/stage1_block_A_semantic.npy"""
    return f"data/features/{stage}_block_{block_name}.npy"


def load_if_exists(path):
    """
    If checkpoint file exists → load and return it (skip recomputing).
    If not             → return None (needs to be computed).
    """
    if os.path.exists(path):
        arr = np.load(path)
        size_mb = arr.nbytes / 1024 / 1024
        print(f"  ✅ CHECKPOINT EXISTS — loaded {path}  shape={arr.shape}  ({size_mb:.1f} MB)")
        return arr
    return None


def save_now(arr, path, label):
    """Save array immediately. Print confirmation with size."""
    np.save(path, arr)
    size_mb = arr.nbytes / 1024 / 1024
    print(f"  💾 SAVED {label} → {path}  shape={arr.shape}  ({size_mb:.1f} MB)")


# ═════════════════════════════════════════════════════════════
# BLOCK A — SEMANTIC EMBEDDINGS (384 dims)
# ═════════════════════════════════════════════════════════════

def run_block_A(texts, stage):
    path = ckpt_path(stage, "A_semantic")
    arr  = load_if_exists(path)
    if arr is not None:
        return arr

    print(f"\n  ── Block A: Semantic Embeddings ({len(texts)} texts) ──")
    t0 = time.time()

    result = embed_model.encode(
        texts,
        batch_size           = 128,       # larger batch = faster on GPU
        normalize_embeddings = True,
        show_progress_bar    = True,
        convert_to_numpy     = True,
    ).astype(np.float32)

    mins = (time.time() - t0) / 60
    print(f"  Block A done in {mins:.1f} min")
    save_now(result, path, "Block A — Semantic (384)")
    return result


# ═════════════════════════════════════════════════════════════
# BLOCK B — STYLOMETRIC FEATURES (10 dims)
# ═════════════════════════════════════════════════════════════

def extract_stylometric(text):
    words = text.split()
    if not words:
        return np.zeros(10, dtype=np.float32)

    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]

    avg_word_len  = np.mean([len(w) for w in words])
    avg_sent_len  = len(words) / max(len(sentences), 1)
    sent_std      = np.std([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0
    unique_ratio  = len(set(words)) / len(words)
    word_counts   = Counter(words)
    hapax         = sum(1 for c in word_counts.values() if c == 1) / len(words)
    stop_ratio    = sum(1 for w in words if w.lower() in stop_words) / len(words)
    punct_density = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
    capital_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    digit_ratio   = sum(1 for c in text if c.isdigit())  / max(len(text), 1)
    special_ratio = sum(1 for c in text if not c.isalnum() and c not in string.whitespace) / max(len(text), 1)

    return np.array([
        avg_word_len, avg_sent_len, sent_std,
        unique_ratio, hapax, stop_ratio,
        punct_density, capital_ratio,
        digit_ratio, special_ratio
    ], dtype=np.float32)


def run_block_B(texts, stage):
    path = ckpt_path(stage, "B_stylometric")
    arr  = load_if_exists(path)
    if arr is not None:
        return arr

    print(f"\n  ── Block B: Stylometric Features ({len(texts)} texts) ──")
    t0     = time.time()
    result = np.array([extract_stylometric(t) for t in tqdm(texts, desc="  Stylometric")],
                      dtype=np.float32)
    print(f"  Block B done in {(time.time()-t0)/60:.1f} min")
    save_now(result, path, "Block B — Stylometric (10)")
    return result


# ═════════════════════════════════════════════════════════════
# BLOCK C — STATISTICAL FEATURES (7 dims)
# ═════════════════════════════════════════════════════════════

def entropy(elements):
    counter = Counter(elements)
    total   = len(elements)
    if total == 0:
        return 0.0
    return -sum((c / total) * np.log2(c / total) for c in counter.values())


def extract_statistical(text):
    words = text.split()
    if not words:
        return np.zeros(7, dtype=np.float32)

    char_entropy    = entropy(list(text))
    word_entropy    = entropy(words)
    token_div       = len(set(words)) / len(words)
    counter         = Counter(words)
    repetition_ratio= sum(1 for c in counter.values() if c > 1) / max(len(counter), 1)
    freqs           = list(counter.values())
    mean_freq       = np.mean(freqs)
    burstiness      = np.var(freqs) / mean_freq if mean_freq > 0 else 0
    M1              = len(words)
    M2              = sum(f ** 2 for f in counter.values())
    yule_k          = 10000 * (M2 - M1) / (M1 * M1) if M1 > 1 else 0
    simpson         = 1 - sum((f / M1) ** 2 for f in counter.values())

    return np.array([
        char_entropy, word_entropy, token_div,
        repetition_ratio, burstiness, yule_k, simpson
    ], dtype=np.float32)


def run_block_C(texts, stage):
    path = ckpt_path(stage, "C_statistical")
    arr  = load_if_exists(path)
    if arr is not None:
        return arr

    print(f"\n  ── Block C: Statistical Features ({len(texts)} texts) ──")
    t0     = time.time()
    result = np.array([extract_statistical(t) for t in tqdm(texts, desc="  Statistical")],
                      dtype=np.float32)
    print(f"  Block C done in {(time.time()-t0)/60:.1f} min")
    save_now(result, path, "Block C — Statistical (7)")
    return result


# ═════════════════════════════════════════════════════════════
# BLOCK D — LINGUISTIC FEATURES (10 dims) — SpaCy
# ═════════════════════════════════════════════════════════════

def extract_linguistic(text):
    doc    = nlp(text[:5000])
    tokens = [t for t in doc if not t.is_space]
    if not tokens:
        return np.zeros(10, dtype=np.float32)

    pos_counts = Counter(t.pos_ for t in tokens)
    total      = len(tokens)

    noun_ratio = pos_counts.get("NOUN", 0) / total
    verb_ratio = pos_counts.get("VERB", 0) / total
    adj_ratio  = pos_counts.get("ADJ",  0) / total
    adv_ratio  = pos_counts.get("ADV",  0) / total
    pron_ratio = pos_counts.get("PRON", 0) / total
    conj_ratio = (pos_counts.get("CCONJ", 0) + pos_counts.get("SCONJ", 0)) / total
    pos_diversity   = len(pos_counts) / 17.0
    dep_complexity  = np.mean([len(list(t.children)) for t in doc]) if tokens else 0

    # Readability (Flesch-Kincaid approximation)
    words = text.split()
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    def syllables(word):
        word  = word.lower()
        count = 0
        prev_v = False
        for ch in word:
            v = ch in "aeiouy"
            if v and not prev_v:
                count += 1
            prev_v = v
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)
    syl_count  = sum(syllables(w) for w in words)
    nw, ns     = max(len(words), 1), max(len(sents), 1)
    fk_score   = 206.835 - 1.015 * (nw / ns) - 84.6 * (syl_count / nw)
    readability = max(0.0, min(100.0, fk_score)) / 100.0

    # Named entity density (lightweight — ner disabled, use heuristic)
    capitalized = sum(1 for w in words if w and w[0].isupper())
    ent_density = capitalized / max(len(words), 1)

    return np.array([
        noun_ratio, verb_ratio, adj_ratio, adv_ratio, pron_ratio,
        conj_ratio, pos_diversity, dep_complexity, ent_density, readability
    ], dtype=np.float32)


def run_block_D(texts, stage):
    path = ckpt_path(stage, "D_linguistic")
    arr  = load_if_exists(path)
    if arr is not None:
        return arr

    print(f"\n  ── Block D: Linguistic Features ({len(texts)} texts) ──")
    t0     = time.time()
    result = np.array([extract_linguistic(t) for t in tqdm(texts, desc="  Linguistic")],
                      dtype=np.float32)
    print(f"  Block D done in {(time.time()-t0)/60:.1f} min")
    save_now(result, path, "Block D — Linguistic (10)")
    return result


# ═════════════════════════════════════════════════════════════
# BLOCK E — PERPLEXITY (1 dim) — GPT-2 on GPU
# ═════════════════════════════════════════════════════════════

def extract_perplexity_batch(texts, batch_size=32):
    """
    Compute GPT-2 perplexity for all texts.
    Runs on GPU if available — much faster than one-by-one.
    Texts truncated to 512 tokens to fit VRAM.
    """
    all_perps = []

    for i in tqdm(range(0, len(texts), batch_size), desc="  Perplexity"):
        batch = texts[i : i + batch_size]

        for text in batch:
            try:
                enc = gpt_tokenizer(
                    text[:1000],
                    return_tensors = "pt",
                    truncation     = True,
                    max_length     = 512,
                )
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                with torch.no_grad():
                    out  = gpt_model(**enc, labels=enc["input_ids"])
                    perp = torch.exp(out.loss).item()
                perp = min(perp, 10000.0)   # cap outliers
            except Exception:
                perp = 500.0                # fallback for edge cases
            all_perps.append(perp)

    return np.array(all_perps, dtype=np.float32)


def run_block_E(texts, stage):
    path = ckpt_path(stage, "E_perplexity")
    arr  = load_if_exists(path)
    if arr is not None:
        return arr

    print(f"\n  ── Block E: Perplexity Features ({len(texts)} texts on {DEVICE}) ──")
    t0     = time.time()
    result = extract_perplexity_batch(texts, batch_size=32)
    print(f"  Block E done in {(time.time()-t0)/60:.1f} min")
    save_now(result, path, "Block E — Perplexity (1)")
    return result


# ═════════════════════════════════════════════════════════════
# BLOCK F — STRUCTURAL FEATURES (8 dims)
# ═════════════════════════════════════════════════════════════

def extract_structural(text):
    words = text.split()
    if not words:
        return np.zeros(8, dtype=np.float32)

    sentences  = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    n_words    = max(len(words), 1)
    n_sents    = max(len(sentences), 1)
    n_chars    = max(len(text), 1)

    imp_starters = {
        "ignore", "tell", "give", "write", "act", "pretend", "forget",
        "bypass", "do", "make", "show", "stop", "start", "roleplay"
    }

    q_density    = text.count("?") / n_chars
    exc_density  = text.count("!") / n_chars
    imp_ratio    = sum(
        1 for s in sentences
        if s.split() and s.split()[0].lower() in imp_starters
    ) / n_sents
    you_ratio    = text.lower().count("you") / n_words
    modal_ratio  = text.lower().count("can")  / n_words
    quote_density= text.count('"') / n_chars
    allcaps_ratio= sum(1 for w in words if w.isupper() and len(w) > 1) / n_words
    sent_var     = np.var([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0

    return np.array([
        q_density, exc_density, imp_ratio, you_ratio,
        modal_ratio, quote_density, allcaps_ratio, sent_var
    ], dtype=np.float32)


def run_block_F(texts, stage):
    path = ckpt_path(stage, "F_structural")
    arr  = load_if_exists(path)
    if arr is not None:
        return arr

    print(f"\n  ── Block F: Structural Features ({len(texts)} texts) ──")
    t0     = time.time()
    result = np.array([extract_structural(t) for t in tqdm(texts, desc="  Structural")],
                      dtype=np.float32)
    print(f"  Block F done in {(time.time()-t0)/60:.1f} min")
    save_now(result, path, "Block F — Structural (8)")
    return result


# ═════════════════════════════════════════════════════════════
# MERGE AND SAVE FINAL (420 dims)
# ═════════════════════════════════════════════════════════════

def merge_and_save(stage, A, B, C, D, E, F, labels):
    """
    Concatenate all 6 blocks → 420-dim matrix.
    Saves to both data/features/ and data/processed/
    """
    E_col = E.reshape(-1, 1) if E.ndim == 1 else E

    X = np.concatenate([A, B, C, D, E_col, F], axis=1)
    assert X.shape[1] == 420, f"Expected 420 features, got {X.shape[1]}"

    # Save to data/features/
    feat_path   = f"data/features/{stage}_features.npy"
    labels_path = f"data/features/{stage}_labels.npy"
    np.save(feat_path,   X)
    np.save(labels_path, labels)
    print(f"\n  💾 FINAL MERGED → {feat_path}  shape={X.shape}")

    # Also save to data/processed/ so model_screening.py finds them
    proc_feat   = f"data/processed/{stage}_features.npy"
    proc_labels = f"data/processed/{stage}_labels.npy"
    np.save(proc_feat,   X)
    np.save(proc_labels, labels)
    print(f"  💾 COPY         → {proc_feat}  shape={X.shape}")

    return X


# ═════════════════════════════════════════════════════════════
# MAIN — runs both stages with full checkpoint protection
# ═════════════════════════════════════════════════════════════

def extract_stage(stage, csv_path):
    print(f"\n{'='*60}")
    print(f"  EXTRACTING: {stage.upper()}")
    print(f"{'='*60}")

    df    = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    labels= df["label"].values.astype(np.int32)

    print(f"  Samples : {len(texts)}")
    print(f"  Labels  : {dict(zip(*np.unique(labels, return_counts=True)))}\n")

    # Save labels checkpoint too
    lbl_path = f"data/features/{stage}_labels.npy"
    if not os.path.exists(lbl_path):
        np.save(lbl_path, labels)
        print(f"  💾 Labels saved → {lbl_path}")

    # ── Run each block (skips if checkpoint exists) ───────────
    A = run_block_A(texts, stage)
    B = run_block_B(texts, stage)
    C = run_block_C(texts, stage)
    D = run_block_D(texts, stage)
    E = run_block_E(texts, stage)
    F = run_block_F(texts, stage)

    # ── Merge all blocks into final 420-dim matrix ────────────
    final_path = f"data/features/{stage}_features.npy"
    if os.path.exists(final_path):
        print(f"\n  ✅ Final features already merged: {final_path}")
        existing = np.load(final_path)
        print(f"     shape={existing.shape}")
    else:
        merge_and_save(stage, A, B, C, D, E, F, labels)

    print(f"\n  ✅ {stage.upper()} COMPLETE\n")


if __name__ == "__main__":
    total_start = time.time()

    extract_stage("stage1", "data/processed/stage1_clean.csv")
    extract_stage("stage2", "data/processed/stage2_clean.csv")

    total_mins = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"  ALL DONE in {total_mins:.1f} minutes")
    print(f"{'='*60}")
    print(f"\n  Files ready for model screening:")
    print(f"    data/processed/stage1_features.npy")
    print(f"    data/processed/stage1_labels.npy")
    print(f"    data/processed/stage2_features.npy")
    print(f"    data/processed/stage2_labels.npy")