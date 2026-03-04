"""
╔══════════════════════════════════════════════════════════════╗
║  STAGE: EXPLORATORY DATA ANALYSIS (EDA)                      ║
║                                                              ║
║  Every single comparison is:                                 ║
║    - Class 0 vs Class 1 side by side                         ║
║    - Shown as a plot                                         ║
║    - Tested for statistical significance (Mann-Whitney U)    ║
║    - Summarized in numbers (saved to CSV)                    ║
║                                                              ║
║  Sections:                                                   ║
║    A. Dataset overview                                       ║
║    B. Text length comparisons                                ║
║    C. Vocabulary & lexical comparisons                       ║
║    D. Punctuation & special character comparisons            ║
║    E. Capitalization comparisons                             ║
║    F. Sentence-level comparisons                             ║
║    G. Word frequency analysis (top 30 per class)             ║
║    H. Entropy comparisons                                    ║
║    I. Structural signal comparisons                          ║
║    J. Jailbreak-specific keyword analysis (Stage 2 only)     ║
║    K. Correlation heatmap of all computed features           ║
║    L. Statistical significance summary table                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from scipy import stats

# ─────────────────────────────────────────────────────────────
# GLOBAL PLOT STYLE
# ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
CLASS_COLORS = {0: "#4C72B0", 1: "#DD8452"}   # blue=class0, orange=class1
ALPHA = 0.70


# ─────────────────────────────────────────────────────────────
# UTILITY: Mann-Whitney U significance test
# Non-parametric — works even when distributions aren't normal
# Returns: U statistic, p-value, interpretation string
# ─────────────────────────────────────────────────────────────
def mann_whitney(vals_0, vals_1):
    """
    Tests whether two groups differ significantly.
    H0: The two distributions are the same.
    H1: They differ.
    p < 0.05 → significant difference → worth noting in thesis.
    """
    u_stat, p_val = stats.mannwhitneyu(vals_0, vals_1, alternative="two-sided")
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    return u_stat, p_val, sig


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


# ─────────────────────────────────────────────────────────────
# FEATURE COMPUTATION
# All features computed once and stored in a dataframe
# so we can run comparisons, significance tests, correlations
# ─────────────────────────────────────────────────────────────

STOPWORDS = {
    "the","a","an","is","in","it","of","to","and","or","for",
    "with","on","at","by","from","that","this","was","are","be",
    "as","he","she","they","we","i","his","her","its","our","their",
    "but","if","not","have","had","has","do","did","does","will",
    "would","could","should","may","might","can","shall","about"
}


def compute_all_features(df):
    """
    Compute all EDA features for every sample.
    Returns df with new columns — used for all comparisons.
    """
    print("  Computing text features for EDA...")
    texts = df["text"].tolist()
    rows = []

    for text in texts:
        words = text.split()
        chars = list(text)
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        n_words = len(words) if words else 1
        n_chars = len(text) if text else 1
        n_sents = len(sentences) if sentences else 1

        # ── Length features ───────────────────────────────────
        char_len   = n_chars
        word_count = n_words
        sent_count = n_sents
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        # ── Vocabulary / lexical features ─────────────────────
        lower_words = [w.lower() for w in words]
        unique_words = set(lower_words)
        freq = Counter(lower_words)
        hapax = sum(1 for w, c in freq.items() if c == 1)

        unique_ratio = len(unique_words) / n_words
        hapax_ratio  = hapax / len(freq) if freq else 0
        stop_count   = sum(1 for w in lower_words if w in STOPWORDS)
        stop_ratio   = stop_count / n_words

        # ── Punctuation features ──────────────────────────────
        punct_chars = sum(1 for c in text if c in '.,;:!?()[]{}"\'-')
        punct_density    = punct_chars / n_chars
        question_count   = text.count("?")
        exclaim_count    = text.count("!")
        comma_count      = text.count(",")
        question_density = question_count / n_sents
        exclaim_density  = exclaim_count  / n_sents
        comma_density    = comma_count    / n_chars

        # ── Capitalization features ───────────────────────────
        alpha_chars = [c for c in text if c.isalpha()]
        cap_ratio   = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if alpha_chars else 0
        upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        allcaps_ratio = upper_words / n_words

        # ── Digit / special char features ─────────────────────
        digit_ratio   = sum(1 for c in text if c.isdigit()) / n_chars
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / n_chars

        # ── Sentence-level features ───────────────────────────
        sent_lens = [len(s.split()) for s in sentences]
        avg_sent_len  = np.mean(sent_lens)  if sent_lens else 0
        sent_len_std  = np.std(sent_lens)   if len(sent_lens) > 1 else 0
        sent_len_var  = np.var(sent_lens)   if len(sent_lens) > 1 else 0

        # ── Entropy features ──────────────────────────────────
        def entropy(items):
            total = len(items)
            if total == 0: return 0
            freq_map = Counter(items)
            return -sum((c/total) * math.log2(c/total) for c in freq_map.values() if c > 0)

        char_entropy = entropy(chars)
        word_entropy = entropy(lower_words)

        # ── Structural / intent signals ───────────────────────
        second_person = {"you","your","yours","yourself","you're","you've","you'll","you'd"}
        modal_verbs   = {"can","could","will","would","shall","should","may","might","must"}
        imperative_starters = {
            "do","dont","don't","please","make","tell","show","give","stop",
            "start","ignore","act","forget","pretend","bypass","jailbreak"
        }

        you_ratio   = sum(1 for w in lower_words if w in second_person)   / n_words
        modal_ratio = sum(1 for w in lower_words if w in modal_verbs)     / n_words
        imp_sents   = sum(1 for s in sentences
                          if s.split() and s.split()[0].lower() in imperative_starters)
        imp_ratio   = imp_sents / n_sents
        quote_density = (text.count('"') + text.count("'")) / n_chars

        # ── Readability (Flesch-Kincaid approx) ──────────────
        def count_syllables(word):
            word = word.lower().rstrip(".,;:!?")
            vowels = "aeiouy"
            count = 0
            prev_v = False
            for ch in word:
                is_v = ch in vowels
                if is_v and not prev_v:
                    count += 1
                prev_v = is_v
            if word.endswith("e") and count > 1:
                count -= 1
            return max(1, count)

        syllables = sum(count_syllables(w) for w in words)
        if sentences and words:
            fk = 206.835 - 1.015*(n_words/n_sents) - 84.6*(syllables/n_words)
            readability = max(0, min(100, fk))
        else:
            readability = 0

        rows.append({
            # length
            "char_len":        char_len,
            "word_count":      word_count,
            "sent_count":      sent_count,
            "avg_word_len":    avg_word_len,
            # lexical
            "unique_ratio":    unique_ratio,
            "hapax_ratio":     hapax_ratio,
            "stop_ratio":      stop_ratio,
            # punctuation
            "punct_density":   punct_density,
            "question_density":question_density,
            "exclaim_density": exclaim_density,
            "comma_density":   comma_density,
            # capitalization
            "cap_ratio":       cap_ratio,
            "allcaps_ratio":   allcaps_ratio,
            # digit/special
            "digit_ratio":     digit_ratio,
            "special_ratio":   special_ratio,
            # sentence
            "avg_sent_len":    avg_sent_len,
            "sent_len_std":    sent_len_std,
            "sent_len_var":    sent_len_var,
            # entropy
            "char_entropy":    char_entropy,
            "word_entropy":    word_entropy,
            # structural
            "you_ratio":       you_ratio,
            "modal_ratio":     modal_ratio,
            "imp_ratio":       imp_ratio,
            "quote_density":   quote_density,
            # readability
            "readability":     readability,
        })

    feat_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)


# ─────────────────────────────────────────────────────────────
# COMPARISON PLOT HELPER
# Produces one figure per feature: histogram + boxplot side by side
# ─────────────────────────────────────────────────────────────
def comparison_plot(df, feature, title, xlabel, label_names, out_path, clip_pct=99):
    """
    For a single feature:
      Left  → Overlapping histogram (Class 0 vs Class 1)
      Right → Side-by-side boxplot (Class 0 vs Class 1)
      Bottom → Mann-Whitney U significance annotation
    """
    vals = {}
    for lbl, name in label_names.items():
        vals[lbl] = df[df["label"] == lbl][feature].dropna().values

    u_stat, p_val, sig = mann_whitney(vals[0], vals[1])

    # Clip outliers for visual clarity (doesn't affect stats)
    clip_val = np.percentile(np.concatenate(list(vals.values())), clip_pct)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"{title}\nMann-Whitney U: p={p_val:.4f} {sig}",
                 fontsize=12, fontweight="bold")

    # Left: histogram overlay
    for lbl, name in label_names.items():
        clipped = np.clip(vals[lbl], 0, clip_val)
        axes[0].hist(clipped, bins=50, alpha=ALPHA,
                     color=CLASS_COLORS[lbl], label=name, density=True, edgecolor="none")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution Comparison")
    axes[0].legend()

    # Add mean lines
    for lbl, name in label_names.items():
        m = np.mean(vals[lbl])
        axes[0].axvline(m, color=CLASS_COLORS[lbl], linestyle="--", linewidth=1.5,
                        label=f"{name} mean={m:.3f}")
    axes[0].legend(fontsize=8)

    # Right: boxplot
    box_data  = [np.clip(vals[lbl], 0, clip_val) for lbl in label_names]
    box_labels = list(label_names.values())
    bp = axes[1].boxplot(box_data, patch_artist=True, notch=False, widths=0.4)
    for patch, lbl in zip(bp["boxes"], label_names):
        patch.set_facecolor(CLASS_COLORS[lbl])
        patch.set_alpha(0.7)
    axes[1].set_xticklabels(box_labels)
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel(xlabel)
    axes[1].set_title("Boxplot Comparison")

    # Annotate means on boxplot
    for i, (lbl, name) in enumerate(label_names.items()):
        m = np.mean(vals[lbl])
        axes[1].text(i+1, m, f" μ={m:.3f}", va="center", fontsize=8, color="black")

    plt.tight_layout()
    save_fig(fig, out_path)

    return {
        "feature": feature,
        f"mean_{list(label_names.values())[0]}": np.mean(vals[0]),
        f"mean_{list(label_names.values())[1]}": np.mean(vals[1]),
        f"std_{list(label_names.values())[0]}":  np.std(vals[0]),
        f"std_{list(label_names.values())[1]}":  np.std(vals[1]),
        "U_statistic": u_stat,
        "p_value":     p_val,
        "significance":sig
    }


# ─────────────────────────────────────────────────────────────
# MASTER EDA FUNCTION
# Runs all sections A–L for one dataset
# ─────────────────────────────────────────────────────────────
def run_full_eda(df, stage_name, out_dir, label_names):
    """
    Parameters
    ----------
    df          : cleaned dataframe with 'text' and 'label' columns
    stage_name  : e.g. "Stage 1 — AI vs Human"
    out_dir     : e.g. "results/stage1/eda"
    label_names : dict like {0: "Human", 1: "AI"}
    """
    os.makedirs(out_dir, exist_ok=True)
    sig_results = []  # collects all significance test results for final table

    print(f"\n{'═'*70}")
    print(f"  EDA: {stage_name}")
    print(f"{'═'*70}")

    # ── Compute all features ──────────────────────────────────
    df = compute_all_features(df)
    print(f"  Feature computation complete. Shape: {df.shape}")


    # ══════════════════════════════════════════════════════════
    # SECTION A — DATASET OVERVIEW
    # ══════════════════════════════════════════════════════════
    print(f"\n  Section A: Dataset Overview")

    # Pie chart + bar chart side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{stage_name} — Dataset Overview", fontsize=13, fontweight="bold")

    vc = df["label"].value_counts().sort_index()
    names  = [label_names[k] for k in vc.index]
    colors = [CLASS_COLORS[k] for k in vc.index]

    axes[0].pie(vc.values, labels=names, colors=colors, autopct="%1.1f%%",
                startangle=90, pctdistance=0.75)
    axes[0].set_title("Class Distribution (Pie)")

    bars = axes[1].bar(names, vc.values, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Class Distribution (Count)")
    axes[1].set_ylabel("Sample Count")
    for bar, val in zip(bars, vc.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + vc.max()*0.01,
                     f"{val:,}", ha="center", fontsize=11)

    save_fig(fig, f"{out_dir}/A_dataset_overview.png")

    # Summary stats table
    overview_rows = []
    for lbl, name in label_names.items():
        sub = df[df["label"] == lbl]
        overview_rows.append({
            "Class": name,
            "N Samples": len(sub),
            "% of Total": f"{len(sub)/len(df)*100:.1f}%",
            "Avg Chars": f"{sub['char_len'].mean():.0f}",
            "Avg Words": f"{sub['word_count'].mean():.0f}",
            "Avg Sents": f"{sub['sent_count'].mean():.0f}",
        })
    overview_df = pd.DataFrame(overview_rows)
    overview_df.to_csv(f"{out_dir}/A_overview_stats.csv", index=False)
    print(overview_df.to_string(index=False))


    # ══════════════════════════════════════════════════════════
    # SECTION B — TEXT LENGTH COMPARISONS
    # ══════════════════════════════════════════════════════════
    print(f"\n  Section B: Text Length Comparisons")

    length_features = [
        ("char_len",    "Character Count",        "Number of characters"),
        ("word_count",  "Word Count",              "Number of words"),
        ("sent_count",  "Sentence Count",          "Number of sentences"),
        ("avg_word_len","Average Word Length",     "Avg characters per word"),
        ("avg_sent_len","Average Sentence Length", "Avg words per sentence"),
        ("sent_len_std","Sentence Length Std Dev", "Variability of sentence lengths"),
    ]

    for feat, title, xlabel in length_features:
        r = comparison_plot(df, feat, f"B: {title}", xlabel, label_names,
                            f"{out_dir}/B_{feat}.png")
        sig_results.append(r)


    # ══════════════════════════════════════════════════════════
    # SECTION C — VOCABULARY & LEXICAL COMPARISONS
    # ══════════════════════════════════════════════════════════
    print(f"\n  Section C: Vocabulary & Lexical")

    lexical_features = [
        ("unique_ratio", "Unique Word Ratio",    "Proportion of unique words"),
        ("hapax_ratio",  "Hapax Legomena Ratio", "Proportion of words appearing once"),
        ("stop_ratio",   "Stopword Ratio",       "Proportion of stopwords"),
    ]

    for feat, title, xlabel in lexical_features:
        r = comparison_plot(df, feat, f"C: {title}", xlabel, label_names,
                            f"{out_dir}/C_{feat}.png")
        sig_results.append(r)

    # Top 30 words per class — side by side bar chart
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"{stage_name} — C: Top 30 Most Frequent Words by Class",
                 fontsize=13, fontweight="bold")

    for i, (lbl, name) in enumerate(label_names.items()):
        texts = df[df["label"] == lbl]["text"].tolist()
        all_words = []
        for t in texts:
            words = re.findall(r'\b[a-z]{3,}\b', t.lower())
            # exclude pure stopwords for readability of the plot
            all_words.extend([w for w in words if w not in STOPWORDS])
        top30 = Counter(all_words).most_common(30)
        words_, counts_ = zip(*top30) if top30 else ([], [])
        axes[i].barh(list(words_)[::-1], list(counts_)[::-1],
                     color=CLASS_COLORS[lbl], alpha=0.8, edgecolor="white")
        axes[i].set_title(f"Class: {name}", fontsize=11)
        axes[i].set_xlabel("Frequency")

    plt.tight_layout()
    save_fig(fig, f"{out_dir}/C_top30_words_per_class.png")


    # ══════════════════════════════════════════════════════════
    # SECTION D — PUNCTUATION COMPARISONS
    # ══════════════════════════════════════════════════════════
    print(f"\n  Section D: Punctuation")

    punct_features = [
        ("punct_density",    "Punctuation Density",    "Punctuation chars / total chars"),
        ("question_density", "Question Mark Density",  "? marks per sentence"),
        ("exclaim_density",  "Exclamation Density",    "! marks per sentence"),
        ("comma_density",    "Comma Density",           "Commas per char"),
        ("quote_density",    "Quotation Mark Density",  "Quote chars per char"),
    ]

    for feat, title, xlabel in punct_features:
        r = comparison_plot(df, feat, f"D: {title}", xlabel, label_names,
                            f"{out_dir}/D_{feat}.png")
        sig_results.append(r)


    # ══════════════════════════════════════════════════════════
    # SECTION E — CAPITALIZATION COMPARISONS
    # ══════════════════════════════════════════════════════════
    print(f"\n  Section E: Capitalization")

    cap_features = [
        ("cap_ratio",     "Capital Letter Ratio", "Uppercase letters / all letters"),
        ("allcaps_ratio", "ALL-CAPS Word Ratio",   "Words in all-caps / all words"),
    ]

    for feat, title, xlabel in cap_features:
        r = comparison_plot(df, feat, f"E: {title}", xlabel, label_names,
                            f"{out_dir}/E_{feat}.png")
        sig_results.append(r)


    # ══════════════════════════════════════════════════════════
    # SECTION F — DIGIT & SPECIAL CHARACTER
    # ══════════════════════════════════════════════════════════
    print(f"\n  Section F: Digits & Special Chars")

    num_features = [
        ("digit_ratio",   "Digit Ratio",         "Digits / total characters"),
        ("special_ratio", "Special Char Ratio",   "Non-alphanumeric / total chars"),
    ]

    for feat, title, xlabel in num_features:
        r = comparison_plot(df, feat, f"F: {title}", xlabel, label_names,
                            f"{out_dir}/F_{feat}.png")
        sig_results.append(r)


    # ══════════════════════════════════════════════════════════
    # SECTION G — ENTROPY COMPARISONS
    # ══════════════════════════════════════════════════════════
    print(f"\n  Section G: Entropy")

    entropy_features = [
        ("char_entropy", "Character Entropy",
         "Shannon entropy over character distribution"),
        ("word_entropy", "Word Entropy",
         "Shannon entropy over word distribution"),
    ]

    for feat, title, xlabel in entropy_features:
        r = comparison_plot(df, feat, f"G: {title}", xlabel, label_names,
                            f"{out_dir}/G_{feat}.png")
        sig_results.append(r)


    # ══════════════════════════════════════════════════════════
    # SECTION H — STRUCTURAL / INTENT SIGNALS
    # ══════════════════════════════════════════════════════════
    print(f"\n  Section H: Structural / Intent Signals")

    struct_features = [
        ("you_ratio",    "Second-Person Pronoun Ratio", "you/your/yours per word"),
        ("modal_ratio",  "Modal Verb Ratio",            "can/will/should/... per word"),
        ("imp_ratio",    "Imperative Sentence Ratio",   "imperative sentences / all sentences"),
        ("readability",  "Readability Score",           "Flesch-Kincaid (0=hard, 100=easy)"),
    ]

    for feat, title, xlabel in struct_features:
        r = comparison_plot(df, feat, f"H: {title}", xlabel, label_names,
                            f"{out_dir}/H_{feat}.png")
        sig_results.append(r)


    # ══════════════════════════════════════════════════════════
    # SECTION I — JAILBREAK-SPECIFIC KEYWORDS (Stage 2 only)
    # ══════════════════════════════════════════════════════════
    if "jailbreak" in stage_name.lower() or "stage2" in out_dir.lower() or "stage_2" in out_dir.lower():
        print(f"\n  Section I: Jailbreak Keyword Analysis")

        keywords = [
            "ignore", "act as", "bypass", "roleplay", "pretend",
            "forget", "disregard", "override", "you are now", "DAN",
            "jailbreak", "do anything now", "no restrictions",
            "without restrictions", "as an ai", "fictional", "hypothetically"
        ]

        kw_data = {}
        for lbl, name in label_names.items():
            texts_lower = df[df["label"] == lbl]["text"].str.lower().tolist()
            n = len(texts_lower)
            kw_data[name] = {
                kw: sum(1 for t in texts_lower if kw in t) / n
                for kw in keywords
            }

        kw_df = pd.DataFrame(kw_data)
        kw_df.index.name = "keyword"
        kw_df.to_csv(f"{out_dir}/I_jailbreak_keywords.csv")

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(keywords))
        width = 0.35
        for i, (lbl, name) in enumerate(label_names.items()):
            offset = (i - 0.5) * width
            vals = [kw_data[name][kw] for kw in keywords]
            bars = ax.bar(x + offset, vals, width, label=name,
                          color=CLASS_COLORS[lbl], alpha=0.8, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(keywords, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Proportion of texts containing keyword")
        ax.set_title(f"{stage_name} — I: Jailbreak Keyword Presence by Class",
                     fontsize=12, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, f"{out_dir}/I_jailbreak_keywords.png")


    # ══════════════════════════════════════════════════════════
    # SECTION J — FEATURE CORRELATION HEATMAP
    # Shows which features are correlated with each other
    # Useful for thesis: explains why dimensionality matters
    # ══════════════════════════════════════════════════════════
    print(f"\n  Section J: Correlation Heatmap")

    feat_cols = [
        "char_len","word_count","sent_count","avg_word_len","avg_sent_len",
        "sent_len_std","unique_ratio","hapax_ratio","stop_ratio",
        "punct_density","question_density","exclaim_density","cap_ratio",
        "allcaps_ratio","digit_ratio","special_ratio",
        "char_entropy","word_entropy","you_ratio","modal_ratio",
        "imp_ratio","readability"
    ]

    corr_matrix = df[feat_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # show lower triangle only
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.3, ax=ax, annot_kws={"size": 7})
    ax.set_title(f"{stage_name} — J: Feature Correlation Matrix",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, f"{out_dir}/J_feature_correlation_heatmap.png")


    # ══════════════════════════════════════════════════════════
    # SECTION K — MULTI-FEATURE COMPARISON DASHBOARD
    # One figure with 9 subplots — good for thesis appendix
    # ══════════════════════════════════════════════════════════
    print(f"\n  Section K: Summary Dashboard")

    dashboard_features = [
        ("word_count",       "Word Count"),
        ("unique_ratio",     "Unique Word Ratio"),
        ("avg_sent_len",     "Avg Sentence Length"),
        ("punct_density",    "Punctuation Density"),
        ("cap_ratio",        "Capitalization Ratio"),
        ("char_entropy",     "Character Entropy"),
        ("word_entropy",     "Word Entropy"),
        ("you_ratio",        "2nd Person Pronoun Ratio"),
        ("readability",      "Readability Score"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f"{stage_name} — K: Feature Summary Dashboard",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, (feat, title) in enumerate(dashboard_features):
        ax = axes[i]
        for lbl, name in label_names.items():
            vals = df[df["label"] == lbl][feat].dropna().values
            clip_val = np.percentile(vals, 99)
            clipped = np.clip(vals, 0, clip_val)
            ax.hist(clipped, bins=40, alpha=ALPHA, density=True,
                    color=CLASS_COLORS[lbl], label=name, edgecolor="none")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_fig(fig, f"{out_dir}/K_summary_dashboard.png")


    # ══════════════════════════════════════════════════════════
    # SECTION L — SIGNIFICANCE SUMMARY TABLE
    # This is your thesis Table 4.x
    # ══════════════════════════════════════════════════════════
    print(f"\n  Section L: Statistical Significance Summary")

    sig_df = pd.DataFrame(sig_results)
    sig_df = sig_df.sort_values("p_value")
    sig_df.to_csv(f"{out_dir}/L_significance_summary.csv", index=False)

    print(f"\n  Statistical Significance Results (sorted by p-value):")
    print(f"  {'Feature':<25} {'p-value':>10} {'Sig':>5}")
    print(f"  {'-'*45}")
    for _, row in sig_df.iterrows():
        print(f"  {row['feature']:<25} {row['p_value']:>10.4f} {row['significance']:>5}")

    n_sig = (sig_df["p_value"] < 0.05).sum()
    print(f"\n  Significant features (p < 0.05): {n_sig} / {len(sig_df)}")
    print(f"  Saved → {out_dir}/L_significance_summary.csv")

    print(f"\n✅ EDA Complete: {stage_name}")
    print(f"   All plots saved to → {out_dir}/")

    return df  # return with features added


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":

    os.makedirs("results/stage1/eda", exist_ok=True)
    os.makedirs("results/stage2/eda", exist_ok=True)

    # ── STAGE 1: AI vs Human ─────────────────────────────────
    print("\nLoading Stage 1...")
    df1 = pd.read_csv("data/processed/stage1_clean.csv")
    run_full_eda(
        df        = df1,
        stage_name= "Stage 1 — AI vs Human",
        out_dir   = "results/stage1/eda",
        label_names= {0: "Human", 1: "AI"}
    )

    # ── STAGE 2: Jailbreak ───────────────────────────────────
    print("\nLoading Stage 2...")
    df2 = pd.read_csv("data/processed/stage2_clean.csv")
    run_full_eda(
        df        = df2,
        stage_name= "Stage 2 — Jailbreak Detection",
        out_dir   = "results/stage2/eda",
        label_names= {0: "Safe", 1: "Jailbreak"}
    )

    print("\n" + "="*70)
    print("ALL EDA COMPLETE")
    print("="*70)
    print("Stage 1 plots → results/stage1/eda/")
    print("Stage 2 plots → results/stage2/eda/")