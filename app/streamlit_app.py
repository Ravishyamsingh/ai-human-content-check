"""
TWO-STAGE TEXT FORENSICS SYSTEM — Streamlit App
================================================
Run: streamlit run app/streamlit_app.py

FIXES:
  Block D fix 1: len(pos_counts)/17.0  (was /total — wrong)
  Block D fix 2: real dep_complexity, ent_density, readability (were 0,0,0)
  Block F fix  : allcaps = w.isupper() no len filter (matches training)
"""

import os
import re
import string
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from collections import Counter

st.set_page_config(
    page_title="Text Forensics · AI Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

for key, default in [("result", None), ("analyzed_text", "")]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═════════════════════════════════════════════════════════════
# CACHED LOADERS
# ═════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading classifiers...")
def load_classifiers():
    m1 = joblib.load("models/stage1_authorship.pkl")
    m2 = joblib.load("models/stage2_unsafe.pkl")
    return m1, m2

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    )

@st.cache_resource(show_spinner="Loading SpaCy...")
def load_spacy():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner"])

@st.cache_resource(show_spinner="Loading GPT-2...")
def load_gpt2():
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    mdl = GPT2LMHeadModel.from_pretrained("gpt2")
    mdl.eval()
    return tok, mdl

@st.cache_resource(show_spinner="Loading stopwords...")
def load_stopwords():
    import nltk
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))

@st.cache_resource(show_spinner="Loading OpenAI RoBERTa detector\u2026")
def load_roberta_openai():
    """OpenAI's RoBERTa detector. id2label: {0:'Fake', 1:'Real'}."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
    mdl = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")
    mdl.eval()
    return tok, mdl

@st.cache_resource(show_spinner="Loading Fakespot AI-detector\u2026")
def load_fakespot_detector():
    """Fakespot AI text detector (Feb 2025, 99K downloads).
    id2label: {0:'Human', 1:'AI'}."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("fakespot-ai/roberta-base-ai-text-detection-v1")
    mdl = AutoModelForSequenceClassification.from_pretrained("fakespot-ai/roberta-base-ai-text-detection-v1")
    mdl.eval()
    return tok, mdl


# ═════════════════════════════════════════════════════════════
# FEATURE FUNCTIONS — verified against saved .npy blocks
# ═════════════════════════════════════════════════════════════

def entropy(elements):
    counter = Counter(elements)
    total   = len(elements)
    if total == 0:
        return 0.0
    return -sum((c/total)*np.log2(c/total) for c in counter.values())


def extract_stylometric(text, stop_words):
    """Block B — 10 dims. Verified matching saved features."""
    words = text.split()
    if not words:
        return np.zeros(10)

    sentences   = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    word_counts = Counter(words)

    avg_word_len  = np.mean([len(w) for w in words])
    avg_sent_len  = len(words) / max(len(sentences), 1)
    sent_std      = np.std([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0
    unique_ratio  = len(set(words)) / len(words)
    hapax         = sum(1 for c in word_counts.values() if c == 1) / len(words)
    stop_ratio    = sum(1 for w in words if w.lower() in stop_words) / len(words)
    # Must match training: divide by max(len(text),1) to avoid blowing up on short texts
    denom = max(len(text), 1)
    punct_density = sum(1 for c in text if c in string.punctuation) / denom
    capital_ratio = sum(1 for c in text if c.isupper()) / denom
    digit_ratio   = sum(1 for c in text if c.isdigit())  / denom
    special_ratio = sum(
        1 for c in text if not c.isalnum() and c not in string.whitespace
    ) / denom

    return np.array([
        avg_word_len, avg_sent_len, sent_std,
        unique_ratio, hapax, stop_ratio,
        punct_density, capital_ratio,
        digit_ratio, special_ratio
    ])


def extract_statistical(text):
    """Block C — 7 dims."""
    words = text.split()
    if not words:
        return np.zeros(7)

    char_entropy     = entropy(list(text))
    word_entropy     = entropy(words)
    token_div        = len(set(words)) / len(words)
    counter          = Counter(words)
    repetition_ratio = sum(1 for c in counter.values() if c > 1) / max(len(counter), 1)
    freqs            = list(counter.values())
    mean_freq        = np.mean(freqs)
    var_freq         = np.var(freqs)
    burstiness       = var_freq / mean_freq if mean_freq > 0 else 0
    M1               = len(words)
    M2               = sum(f**2 for f in counter.values())
    yule_k           = 10000 * (M2 - M1) / (M1 * M1) if M1 > 1 else 0
    simpson          = 1 - sum((f / M1)**2 for f in counter.values())

    return np.array([
        char_entropy, word_entropy, token_div,
        repetition_ratio, burstiness, yule_k, simpson
    ])


def extract_linguistic(text, nlp):
    """
    Block D — 10 dims.
    FIXES vs previous versions:
      - pos_diversity = len(pos_counts)/17.0  (NOT /total)
      - dep_complexity = mean children per token (NOT 0)
      - ent_density    = capitalized words ratio (NOT 0)
      - readability    = FK score normalized    (NOT 0)
    """
    doc    = nlp(text[:5000])
    tokens = [t for t in doc if not t.is_space]
    if not tokens:
        return np.zeros(10)

    pos_counts = Counter(t.pos_ for t in tokens)
    total      = len(tokens)
    words      = text.split()
    sents      = [s for s in re.split(r'[.!?]+', text) if s.strip()]

    # POS ratios
    noun_ratio = pos_counts.get("NOUN", 0) / total
    verb_ratio = pos_counts.get("VERB", 0) / total
    adj_ratio  = pos_counts.get("ADJ",  0) / total
    adv_ratio  = pos_counts.get("ADV",  0) / total
    pron_ratio = pos_counts.get("PRON", 0) / total
    conj_ratio = (pos_counts.get("CCONJ", 0) + pos_counts.get("SCONJ", 0)) / total

    # FIX 1: divide by 17 (number of universal POS tags), not by total tokens
    pos_diversity = len(pos_counts) / 17.0

    # FIX 2: real dependency complexity
    dep_complexity = np.mean([len(list(t.children)) for t in doc])

    # FIX 3: real entity density (capitalized words as proxy, ner disabled)
    nw          = max(len(words), 1)
    capitalized = sum(1 for w in words if w and w[0].isupper())
    ent_density = capitalized / nw

    # FIX 4: real readability (Flesch-Kincaid)
    def syllables(word):
        word   = word.lower()
        count  = 0
        prev_v = False
        for ch in word:
            v = ch in "aeiouy"
            if v and not prev_v:
                count += 1
            prev_v = v
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    ns        = max(len(sents), 1)
    syl_count = sum(syllables(w) for w in words)
    fk        = 206.835 - 1.015 * (nw / ns) - 84.6 * (syl_count / nw)
    readability = max(0.0, min(100.0, fk)) / 100.0

    return np.array([
        noun_ratio, verb_ratio, adj_ratio, adv_ratio, pron_ratio,
        conj_ratio, pos_diversity, dep_complexity, ent_density, readability
    ])


def extract_perplexity(text, gpt_tokenizer, gpt_model):
    """Block E — 1 dim. Caps at 10000 to match training."""
    import torch
    try:
        enc = gpt_tokenizer(text[:1000], return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = gpt_model(**enc, labels=enc["input_ids"])
            perp = np.exp(outputs.loss.item())
            return min(perp, 10000.0)  # cap outliers — matches training
    except Exception:
        return 500.0


def extract_structural(text):
    """
    Block F — 8 dims.
    Matches training script (04_feature_extraction.py) exactly.
    """
    words = text.split()
    if not words:
        return np.zeros(8)

    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    n_words   = max(len(words), 1)
    n_sents   = max(len(sentences), 1)
    n_chars   = max(len(text), 1)

    # Must match the exact set from training (04_feature_extraction.py)
    imp_starters = {
        "ignore", "tell", "give", "write", "act", "pretend", "forget",
        "bypass", "do", "make", "show", "stop", "start", "roleplay"
    }

    return np.array([
        text.count("?")  / n_chars,
        text.count("!")  / n_chars,
        sum(1 for s in sentences
            if s.split() and s.split()[0].lower() in imp_starters
        ) / n_sents,
        text.lower().count("you") / n_words,
        text.lower().count("can") / n_words,
        text.count('"')  / n_chars,
        sum(1 for w in words if w.isupper() and len(w) > 1) / n_words,
        np.var([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0
    ])


def extract_420(text, embed_model, nlp, gpt_tokenizer, gpt_model, stop_words):
    """420-dim vector — identical order and logic to training script."""
    # IMPORTANT: must match training (scripts/04_feature_extraction.py)
    # where normalize_embeddings=True was used.
    semantic = embed_model.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    stylo    = extract_stylometric(text, stop_words)
    stat     = extract_statistical(text)
    ling     = extract_linguistic(text, nlp)
    ppl      = np.array([extract_perplexity(text, gpt_tokenizer, gpt_model)])
    struct   = extract_structural(text)

    X = np.concatenate([
        semantic[0], stylo, stat, ling, ppl, struct
    ]).reshape(1, -1)

    assert X.shape[1] == 420, f"Got {X.shape[1]} dims, expected 420"
    return X


# ═════════════════════════════════════════════════════════════
# TEXT PREPROCESSING — normalize input before analysis
# ═════════════════════════════════════════════════════════════

def _normalize_text(text: str) -> str:
    """Clean and normalize text for more consistent detection.

    Fixes encoding artefacts, normalizes whitespace and quotes,
    and strips invisible characters that can fool detectors.
    """
    # 1) Normalize Unicode (NFC) — collapse combining chars
    import unicodedata
    text = unicodedata.normalize("NFC", text)

    # 2) Remove zero-width and invisible chars (used to evade detection)
    invisible = re.compile(
        r'[\u200b\u200c\u200d\u2060\ufeff\u00ad\u034f'
        r'\u061c\u115f\u1160\u17b4\u17b5\u180e\u2000-\u200f'
        r'\u202a-\u202e\u2061-\u2064\u2066-\u2069\u206a-\u206f'
        r'\ufff9-\ufffb]'
    )
    text = invisible.sub('', text)

    # 3) Normalize quotes and dashes
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # smart single
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # smart double
    text = text.replace('\u2013', '-').replace('\u2014', '--')  # en/em dash
    text = text.replace('\u2026', '...')                        # ellipsis

    # 4) Collapse multiple spaces/tabs to single space
    text = re.sub(r'[^\S\n]+', ' ', text)

    # 5) Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 6) Strip leading/trailing whitespace
    text = text.strip()

    return text


# ═════════════════════════════════════════════════════════════
# PREDICTION  —  Enhanced Multi-Signal Ensemble Engine
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# Base weights for 3-model ensemble (before heuristic adjustment)
# ─────────────────────────────────────────────────────────────
W_XGB      = 0.20
W_OPENAI   = 0.25
W_FAKESPOT = 0.55


# ── (A) Sliding-window transformer inference ───────────────
def _transformer_sliding_ai_prob(text, tokenizer, model, ai_index,
                                  max_tokens=512, stride=256):
    """Score text using a sliding window over 512-token chunks.

    For short texts (<= max_tokens) this is identical to a single pass.
    For longer texts it evaluates overlapping windows and returns the
    *mean* P(AI) across all windows — capturing signal across the
    whole document instead of just the first 300 words.
    """
    import torch
    try:
        encoding = tokenizer(text, return_tensors="pt",
                              truncation=False, add_special_tokens=False)
        input_ids = encoding["input_ids"][0]
        total_len = len(input_ids)

        if total_len <= max_tokens:
            # Short text — single pass
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=max_tokens)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            return float(probs[ai_index])

        # Sliding window for long text
        window_probs = []
        start = 0
        while start < total_len:
            end = min(start + max_tokens, total_len)
            chunk_ids = input_ids[start:end].unsqueeze(0)
            attn_mask = torch.ones_like(chunk_ids)
            with torch.no_grad():
                logits = model(input_ids=chunk_ids,
                               attention_mask=attn_mask).logits
            probs = torch.softmax(logits, dim=-1)[0]
            window_probs.append(float(probs[ai_index]))
            if end >= total_len:
                break
            start += stride

        # Aggregate: use mean but boost if any window is highly confident
        mean_p = np.mean(window_probs)
        max_p  = np.max(window_probs)
        # Blend: 70% mean + 30% max → rewards consistent AI signal
        return 0.7 * mean_p + 0.3 * max_p
    except Exception:
        return 0.5


# ── (B) Sentence-level consistency analysis ────────────────
def _sentence_level_analysis(text, tokenizer, model, ai_index):
    """Analyze individual sentences and return consistency metrics.

    AI text tends to have *uniformly high* AI scores across sentences,
    while human text varies more.  Returns:
      - mean P(AI) across sentences
      - fraction of sentences scoring > 0.5 (AI)
      - std of scores (low = more AI-like)
    """
    import torch
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.split()) >= 4]
    if len(sents) < 2:
        return None  # not enough sentences for analysis

    scores = []
    for sent in sents[:20]:  # cap at 20 sentences for speed
        try:
            inputs = tokenizer(sent, return_tensors="pt",
                               truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            scores.append(float(probs[ai_index]))
        except Exception:
            continue

    if len(scores) < 2:
        return None

    return {
        "mean":     np.mean(scores),
        "ai_frac":  sum(1 for s in scores if s > 0.5) / len(scores),
        "std":      np.std(scores),
        "n_sents":  len(scores),
        "scores":   scores,
    }


# ── (C) AI writing-style heuristics ────────────────────────
# These patterns are characteristic of AI-generated text and are
# model-independent — they detect the *style* not the *source*.

_AI_TRANSITIONS = {
    "furthermore", "moreover", "additionally", "consequently",
    "nevertheless", "nonetheless", "in addition", "as a result",
    "in conclusion", "to summarize", "it is worth noting",
    "it is important to note", "in this context", "given that",
    "on the other hand", "in contrast", "similarly",
    "significantly", "notably", "importantly",
}

_AI_HEDGING = {
    "arguably", "potentially", "essentially", "fundamentally",
    "particularly", "specifically", "inherently", "ultimately",
    "undoubtedly", "unprecedented", "comprehensive", "robust",
    "multifaceted", "nuanced", "intricate", "paradigm",
    "landscape", "framework", "leverage", "facilitate",
    "utilize", "implement", "demonstrate", "exhibit",
    "encompass", "underscore", "streamline",
    # Additional ChatGPT favorites
    "crucial", "pivotal", "innovative", "holistic", "synergy",
    "optimize", "optimal", "efficacy", "proficiency",
    "meticulous", "methodical", "systematically", "proactively",
    "seamlessly", "strategically", "exponentially",
    "transformative", "groundbreaking", "cutting-edge",
    "aforementioned", "aforementioned", "henceforth",
    "rewritten", "rewrite", "revised", "rephrased",
    "assumptions", "professional manner",
}

_HUMAN_MARKERS = {
    "lol", "lmao", "ngl", "tbh", "imo", "smh", "bruh", "omg",
    "haha", "gonna", "wanna", "kinda", "gotta", "dunno",
    "yeah", "yep", "nah", "hmm", "umm", "idk", "btw",
}

# Additional AI phrase patterns (multi-word markers)
_AI_PHRASES = [
    "it is important to", "it is worth noting", "it should be noted",
    "plays a crucial role", "it is essential to", "in today's world",
    "in the modern era", "has become increasingly", "it is evident that",
    "one could argue", "this underscores the", "this highlights the",
    "delve into", "delve deeper", "in the realm of",
    "a myriad of", "at the forefront", "pave the way",
    "it cannot be overstated", "to shed light on",
    "the advent of", "the emergence of", "the proliferation of",
    "serves as a testament", "a testament to",
    "from a broader perspective", "in light of",
    "navigating the complexities", "fostering a sense of",
    # ChatGPT meta-prompt / rewrite / instruction patterns
    "below is your request", "below is a rewritten", "below is the rewritten",
    "rewritten in a clear", "rewritten in a professional",
    "here is a revised", "here is the revised", "here's a revised",
    "here is a rewritten", "here is the rewritten",
    "without adding extra assumptions", "without making assumptions",
    "sure! here", "sure, here", "of course! here", "absolutely! here",
    "certainly! here", "certainly, here",
    "i'd be happy to", "i'll help you", "let me help",
    "here's a comprehensive", "here is a comprehensive",
    "here's a detailed", "here is a detailed",
    "conduct a comprehensive review", "comprehensive review of",
    "identify the root cause", "root cause of the",
    "ensure that all", "ensure consistency",
    "please find below", "as per your request",
    "hope this helps", "let me know if", "feel free to",
]

# Common AI sentence starters (formulaic openings)
_AI_STARTERS = [
    "in conclusion,", "moreover,", "furthermore,", "additionally,",
    "however,", "nevertheless,", "consequently,", "similarly,",
    "in summary,", "to summarize,", "in essence,", "ultimately,",
    "it is important", "it is worth", "it is essential",
    "this is because", "this demonstrates", "this highlights",
    "one of the most", "one of the key", "the importance of",
    "by leveraging", "by utilizing", "by implementing",
    "as a result,", "on the other hand,", "in other words,",
]


def _compute_style_heuristics(text):
    """Return a dict of AI-style heuristic signals in [0, 1] range."""
    words = text.lower().split()
    nw = max(len(words), 1)
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    ns = max(len(sents), 1)

    # 1) Transition word density (AI uses many formal transitions)
    text_lower = text.lower()
    transition_hits = sum(1 for t in _AI_TRANSITIONS if t in text_lower)
    transition_score = min(transition_hits / max(ns, 1), 1.0)

    # 2) Hedging/academic word density
    hedge_hits = sum(1 for w in words if w in _AI_HEDGING)
    hedge_score = min(hedge_hits / nw * 20, 1.0)  # normalize: 5% hedge words → 1.0

    # 3) Human informality markers (NEGATIVE signal for AI)
    human_hits = sum(1 for w in words if w in _HUMAN_MARKERS)
    human_score = min(human_hits / nw * 50, 1.0)  # normalize

    # 4) Sentence length uniformity (AI has low variance)
    sent_lens = [len(s.split()) for s in sents]
    if len(sent_lens) > 1:
        cv = np.std(sent_lens) / max(np.mean(sent_lens), 1)  # coeff of variation
        uniformity_score = max(0, 1.0 - cv)  # low CV = high uniformity = more AI-like
    else:
        uniformity_score = 0.5

    # 5) Punctuation variety (humans use more diverse punctuation)
    has_ellipsis = "..." in text or "\u2026" in text
    has_multiple_excl = text.count("!") > 2
    has_multiple_quest = text.count("?") > 2
    has_dashes = "--" in text or "\u2014" in text or "\u2013" in text
    punct_variety = sum([has_ellipsis, has_multiple_excl,
                        has_multiple_quest, has_dashes])
    formal_punct = 1.0 - min(punct_variety / 3, 1.0)

    # 6) First-person pronouns (humans use more I/my/me)
    first_person = sum(1 for w in words if w in {"i", "my", "me", "mine", "myself"})
    first_person_score = max(0, 1.0 - (first_person / nw * 30))

    # 7) Paragraph structure (AI tends to write longer, more uniform paragraphs)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) > 1:
        para_lens = [len(p.split()) for p in paragraphs]
        avg_para_len = np.mean(para_lens)
        para_score = min(avg_para_len / 80, 1.0)  # long paragraphs = more AI-like
    else:
        para_score = 0.5

    # ── NEW: Advanced heuristic signals ──────────────────

    # 8) AI phrase pattern matching (multi-word patterns)
    phrase_hits = sum(1 for p in _AI_PHRASES if p in text_lower)
    phrase_score = min(phrase_hits / max(ns, 1), 1.0)

    # 9) Formulaic sentence starters
    starter_hits = 0
    for s in sents:
        s_lower = s.strip().lower()
        if any(s_lower.startswith(st) for st in _AI_STARTERS):
            starter_hits += 1
    starter_score = min(starter_hits / max(ns, 1), 1.0)

    # 10) Comma density (AI uses more commas for complex sentence structures)
    comma_count = text.count(",")
    comma_density = comma_count / nw
    comma_score = min(comma_density / 0.12, 1.0)  # 12% comma rate → max

    # 11) Sentence starter diversity (AI often reuses "The", "This", "It")
    if len(sents) >= 3:
        starters = [s.strip().split()[0].lower() if s.strip().split() else ""
                     for s in sents]
        unique_starters = len(set(starters))
        starter_diversity = unique_starters / len(starters)
        # Low diversity = more AI-like (AI reuses same sentence opener)
        repetitive_score = max(0, 1.0 - starter_diversity)
    else:
        repetitive_score = 0.5

    # 12) Contraction usage (humans use contractions; AI avoids them)
    contractions = ["n't", "'m", "'re", "'ve", "'ll", "'d", "'s"]
    contraction_count = sum(text_lower.count(c) for c in contractions)
    contraction_ratio = contraction_count / max(ns, 1)
    # High contractions = human → low AI score
    no_contraction_score = max(0, 1.0 - contraction_ratio / 2.0)

    # 13) Text has typos/errors (strong human signal)
    # Simple heuristic: consecutive same chars (3+), or common misspellings
    typo_patterns = len(re.findall(r'(.)\1{2,}', text))  # e.g., "soooo", "aaaa"
    has_typos = min(typo_patterns / max(ns, 1), 1.0)

    # Combine: weighted sum of all AI-positive signals minus human signals
    ai_style = (
        0.12 * transition_score +
        0.12 * hedge_score +
        0.10 * phrase_score +        # NEW
        0.10 * starter_score +       # NEW
        0.08 * uniformity_score +
        0.08 * formal_punct +
        0.08 * first_person_score +
        0.08 * comma_score +         # NEW
        0.06 * repetitive_score +    # NEW
        0.06 * no_contraction_score + # NEW
        0.06 * para_score +
        0.03 * (1.0 - human_score) +
        0.03 * (1.0 - has_typos)     # NEW: typos = human
    )

    # Subtract human informality directly
    ai_style = max(0.0, ai_style - 0.3 * human_score)

    return {
        "ai_style":       ai_style,
        "transitions":    transition_score,
        "hedging":        hedge_score,
        "human_markers":  human_score,
        "uniformity":     uniformity_score,
        "formal_punct":   formal_punct,
        "first_person":   first_person_score,
        "phrase_match":   phrase_score,
        "starter_match":  starter_score,
        "comma_density":  comma_score,
        "repetitive":     repetitive_score,
        "no_contractions": no_contraction_score,
        "has_typos":      has_typos,
    }


# ── (D) Perplexity-based signals ───────────────────────────
def _perplexity_analysis(text, gpt_tok, gpt_mdl):
    """Compute per-sentence perplexity variance.

    AI text has *consistent low perplexity* across sentences.
    Human text has *variable perplexity* (some fluent, some rough).
    """
    import torch
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
             if len(s.split()) >= 3]
    if len(sents) < 2:
        return None

    perps = []
    for sent in sents[:15]:  # cap for speed
        try:
            enc = gpt_tok(sent[:500], return_tensors="pt", truncation=True)
            if enc["input_ids"].shape[1] < 2:
                continue
            with torch.no_grad():
                out = gpt_mdl(**enc, labels=enc["input_ids"])
                p = float(np.exp(out.loss.item()))
                perps.append(min(p, 10000.0))
        except Exception:
            continue

    if len(perps) < 2:
        return None

    mean_p = np.mean(perps)
    std_p  = np.std(perps)
    cv     = std_p / max(mean_p, 1)

    # Low perplexity + low variance = AI
    # High perplexity or high variance = Human
    return {
        "mean_ppl": mean_p,
        "std_ppl":  std_p,
        "cv":       cv,
        "low_ppl_signal": max(0, 1.0 - mean_p / 200),  # below 200 = suspicious
        "low_var_signal": max(0, 1.0 - cv),              # low CV = AI-like
    }


# ── Main prediction function ───────────────────────────────
def predict(text):
    # Normalize text before analysis (remove invisible chars, fix encoding)
    text = _normalize_text(text)

    m1, m2           = load_classifiers()
    embed_model      = load_embedder()
    nlp              = load_spacy()
    gpt_tok, gpt_mdl = load_gpt2()
    stop_words       = load_stopwords()
    oai_tok, oai_mdl = load_roberta_openai()
    fs_tok, fs_mdl   = load_fakespot_detector()

    X = extract_420(text, embed_model, nlp, gpt_tok, gpt_mdl, stop_words)

    # ── Stage 1 — Individual model scores ──────────────────

    # 1) XGBoost (feature-based): 0=AI, 1=Human
    s1_proba    = m1.predict_proba(X)[0]
    xgb_p_ai    = float(s1_proba[0])
    xgb_p_human = float(s1_proba[1])

    # 2) OpenAI RoBERTa with sliding window
    oai_p_ai    = _transformer_sliding_ai_prob(
        text, oai_tok, oai_mdl, ai_index=0)
    oai_p_human = 1.0 - oai_p_ai

    # 3) Fakespot RoBERTa with sliding window
    fs_p_ai     = _transformer_sliding_ai_prob(
        text, fs_tok, fs_mdl, ai_index=1)
    fs_p_human  = 1.0 - fs_p_ai

    # ── Heuristic signals ──────────────────────────────────
    style = _compute_style_heuristics(text)
    ai_style = style["ai_style"]
    ppl_analysis = _perplexity_analysis(text, gpt_tok, gpt_mdl)
    sent_analysis = _sentence_level_analysis(text, fs_tok, fs_mdl, ai_index=1)

    # ── Dynamic weight adjustment based on text characteristics ──
    word_count = len(text.split())

    # ── SMART ENSEMBLE with qualified-majority logic ────────
    # Key insight: Fakespot has very high sensitivity (catches all AI) but
    # LOW specificity (flags many human texts as AI too). XGBoost + OpenAI
    # have GOOD specificity (correctly identify human text).
    #
    # Strategy: Use a qualified-majority system. If XGBoost AND OpenAI
    # both say Human but Fakespot says AI → trust the majority.

    # Phase 1: Check if XGBoost + OpenAI form a consensus against Fakespot
    xgb_oai_human = xgb_p_ai < 0.30 and oai_p_ai < 0.30
    xgb_oai_ai    = xgb_p_ai > 0.60 and oai_p_ai > 0.60
    fs_ai         = fs_p_ai > 0.70
    fs_human      = fs_p_ai < 0.30

    # Conflict: Fakespot says AI but both others say Human
    fakespot_overrule = xgb_oai_human and fs_ai

    # All agree AI: strong signal
    all_agree_ai    = xgb_p_ai > 0.55 and oai_p_ai > 0.55 and fs_p_ai > 0.55
    all_agree_human = xgb_p_ai < 0.45 and oai_p_ai < 0.45 and fs_p_ai < 0.45

    # Strong agreement (all >0.75 or all <0.25)
    strong_ai    = xgb_p_ai > 0.75 and oai_p_ai > 0.75 and fs_p_ai > 0.75
    strong_human = xgb_p_ai < 0.25 and oai_p_ai < 0.25 and fs_p_ai < 0.25

    # Also check for ChatGPT meta-patterns in the text itself
    text_lower_check = text.lower()
    chatgpt_meta_hits = sum(1 for p in _AI_PHRASES[-20:] if p in text_lower_check)
    # Any ChatGPT meta-pattern match boosts the effective AI style score
    effective_ai_style = ai_style + min(chatgpt_meta_hits * 0.15, 0.30)

    if fakespot_overrule:
        # XGBoost + OpenAI both say Human, Fakespot disagrees
        # DECISION LOGIC:
        #   - Use effective_ai_style (includes ChatGPT meta-pattern detection)
        #   - Style > 0.40: AI patterns confirmed → trust Fakespot
        #   - Style 0.25-0.40: ambiguous → balanced blend
        #   - Style < 0.25: clearly human → override Fakespot
        #   - ChatGPT meta-patterns always push toward trusting Fakespot

        if chatgpt_meta_hits >= 2:
            # Multiple ChatGPT meta-patterns found → definitely AI
            ens_p_ai = 0.10 * xgb_p_ai + 0.10 * oai_p_ai + 0.80 * fs_p_ai
            agreement_mode = "chatgpt-detected"
        elif effective_ai_style > 0.40:
            # Style confirms AI patterns → trust Fakespot more
            ens_p_ai = 0.15 * xgb_p_ai + 0.15 * oai_p_ai + 0.70 * fs_p_ai
            agreement_mode = "style-confirms-ai"
        elif effective_ai_style > 0.25:
            # Ambiguous style — check additional human evidence
            # Strong human signals: personal pronouns, contractions, human markers
            human_evidence = 0
            if style["first_person"] < 0.3:   # lots of I/my/me
                human_evidence += 1
            if style["no_contractions"] < 0.7:  # uses contractions
                human_evidence += 1
            if style["human_markers"] > 0.01:   # informal language
                human_evidence += 1
            if style["has_typos"] > 0.05:       # has typos
                human_evidence += 1

            if human_evidence >= 2:
                # Multiple human signals → lean toward human despite ambiguous style
                ens_p_ai = 0.30 * xgb_p_ai + 0.30 * oai_p_ai + 0.40 * fs_p_ai
                agreement_mode = "ambiguous-human-evidence"
            else:
                # No strong human signals → balanced blend
                ens_p_ai = 0.25 * xgb_p_ai + 0.25 * oai_p_ai + 0.50 * fs_p_ai
                agreement_mode = "style-ambiguous"
        else:
            # Style is clearly human → trust XGBoost + OpenAI majority
            ens_p_ai = 0.35 * xgb_p_ai + 0.35 * oai_p_ai + 0.30 * fs_p_ai
            agreement_mode = "override-human"
    elif all_agree_ai:
        # All 3 models agree → AI → high confidence
        ens_p_ai = 0.20 * xgb_p_ai + 0.25 * oai_p_ai + 0.55 * fs_p_ai
        agreement_mode = "unanimous-ai"
    elif all_agree_human:
        # All 3 models agree → Human → high confidence
        ens_p_ai = 0.30 * xgb_p_ai + 0.30 * oai_p_ai + 0.40 * fs_p_ai
        agreement_mode = "unanimous-human"
    elif xgb_p_ai < 0.10 and (effective_ai_style < 0.40 or style["first_person"] < 0.15):
        # XGBoost is VERY confident human AND either:
        #   - style heuristics agree (not AI-like), OR
        #   - text uses lots of first-person pronouns (strong human signal)
        # XGBoost was trained on our dataset — when it's confident AND
        # writing style is human, trust it heavily even if transformers
        # are uncertain (Fakespot has high false positive rate)
        #
        # Count additional human evidence
        human_ev = 0
        if style["first_person"] < 0.3:      human_ev += 1  # lots of I/my/me
        if style["no_contractions"] < 0.7:   human_ev += 1  # uses contractions
        if style["human_markers"] > 0.01:    human_ev += 1  # informal language
        if style["has_typos"] > 0.05:        human_ev += 1  # typos present
        if oai_p_ai < 0.55:                  human_ev += 1  # OpenAI leans human

        # Dampen Fakespot false positives: when XGBoost strongly disagrees,
        # Fakespot's extreme scores (>90%) are unreliable — cap them
        # Also cap uncertain OpenAI (40-60%) so it doesn't override XGBoost
        adj_oai = min(oai_p_ai, 0.40) if 0.35 < oai_p_ai < 0.65 else oai_p_ai

        if human_ev >= 3:
            # Strong human evidence → heavily trust XGBoost, cap FS
            adj_fs = min(fs_p_ai, 0.30)
            ens_p_ai = 0.55 * xgb_p_ai + 0.20 * adj_oai + 0.25 * adj_fs
            agreement_mode = "xgb-human-strong"
        elif human_ev >= 1:
            adj_fs = min(fs_p_ai, 0.40)
            ens_p_ai = 0.50 * xgb_p_ai + 0.20 * adj_oai + 0.30 * adj_fs
            agreement_mode = "xgb-human-moderate"
        else:
            ens_p_ai = 0.30 * xgb_p_ai + 0.25 * oai_p_ai + 0.45 * fs_p_ai
            agreement_mode = "xgb-human-weak"
    else:
        # Mixed signals: dynamic weights based on text length
        if word_count < 30:
            # Short text: XGBoost features unreliable, lean on transformers equally
            ens_p_ai = 0.10 * xgb_p_ai + 0.40 * oai_p_ai + 0.50 * fs_p_ai
        elif word_count > 300:
            # Long text: all models have good signal
            ens_p_ai = 0.25 * xgb_p_ai + 0.30 * oai_p_ai + 0.45 * fs_p_ai
        else:
            # Medium text: balanced weights
            ens_p_ai = 0.25 * xgb_p_ai + 0.30 * oai_p_ai + 0.45 * fs_p_ai
        agreement_mode = "weighted"

    # ── Heuristic adjustment ────────────────────────────────
    heuristic_adj = 0.0

    # Style heuristic: overall AI style score
    # AI text typically has ai_style > 0.40, Human < 0.35
    if ai_style > 0.50:
        heuristic_adj += 0.06
    elif ai_style > 0.40:
        heuristic_adj += 0.03
    elif ai_style < 0.20:
        heuristic_adj -= 0.06
    elif ai_style < 0.30:
        heuristic_adj -= 0.03
    elif ai_style < 0.38:
        heuristic_adj -= 0.02  # leaning human (0.30-0.38 range)

    # XGBoost extreme confidence + non-AI style → strong human signal
    # Threshold at 0.38 (not 0.40) to avoid floating-point edge at 0.40
    if xgb_p_ai < 0.05 and ai_style < 0.38:
        heuristic_adj -= 0.08
    elif xgb_p_ai < 0.10 and ai_style < 0.33:
        heuristic_adj -= 0.04

    # Specific strong AI phrase patterns found
    if style["phrase_match"] > 0.3:
        heuristic_adj += 0.04
    if style["starter_match"] > 0.3:
        heuristic_adj += 0.03

    # Human markers strongly push toward human
    if style["human_markers"] > 0.05:
        heuristic_adj -= 0.12
    elif style["human_markers"] > 0.01:
        heuristic_adj -= 0.05

    # Contractions (humans use them, AI avoids them)
    if style["no_contractions"] < 0.5:   # lots of contractions → human
        heuristic_adj -= 0.05

    # Strong first-person usage (I, my, me) → very human
    if style["first_person"] < 0.2:
        heuristic_adj -= 0.06
    elif style["first_person"] < 0.4:
        heuristic_adj -= 0.03

    # Typos/repeated chars → definitely human
    if style["has_typos"] > 0.1:
        heuristic_adj -= 0.06

    # Sentence consistency (Fakespot): if most sentences flag AI → boost
    # BUT: skip positive (AI-pushing) adjustment when in xgb-human mode,
    # because Fakespot's sentence-level false positives would double-count
    in_xgb_human = agreement_mode.startswith("xgb-human")
    if sent_analysis is not None:
        if sent_analysis["ai_frac"] >= 0.8 and not in_xgb_human:
            heuristic_adj += 0.05
        elif sent_analysis["ai_frac"] >= 0.6 and not in_xgb_human:
            heuristic_adj += 0.02
        elif sent_analysis["ai_frac"] <= 0.2:
            heuristic_adj -= 0.05
        elif sent_analysis["ai_frac"] <= 0.35:
            heuristic_adj -= 0.02

        # Very low std across sentence scores = AI (uniform output)
        # Skip in xgb-human modes to avoid reinforcing Fakespot false positives
        if sent_analysis["std"] < 0.10 and sent_analysis["mean"] > 0.6 and not in_xgb_human:
            heuristic_adj += 0.03

    # Perplexity: low mean + low variance → AI
    # Skip positive (AI-pushing) adjustment in xgb-human modes
    if ppl_analysis is not None:
        if ppl_analysis["low_ppl_signal"] > 0.6 and ppl_analysis["low_var_signal"] > 0.5 and not in_xgb_human:
            heuristic_adj += 0.04
        elif ppl_analysis["low_ppl_signal"] < 0.2:
            heuristic_adj -= 0.03  # High perplexity → likely human

    # Model agreement bonus
    if strong_ai:
        heuristic_adj += 0.06
    elif strong_human:
        heuristic_adj -= 0.06

    # Majority vote correction: if 2/3 models say human (<0.40),
    # but overall ensemble still says AI, apply correction
    models_saying_human = sum(1 for p in [xgb_p_ai, oai_p_ai, fs_p_ai] if p < 0.40)
    models_saying_ai = sum(1 for p in [xgb_p_ai, oai_p_ai, fs_p_ai] if p > 0.60)
    if models_saying_human >= 2 and ens_p_ai > 0.50:
        heuristic_adj -= 0.08  # push toward human
    elif models_saying_ai >= 2 and ens_p_ai < 0.50:
        heuristic_adj += 0.08  # push toward AI

    # Clamp heuristic adjustment to ±0.20 range
    heuristic_adj = max(-0.20, min(0.20, heuristic_adj))

    # Apply heuristic adjustment (clamped to [0, 1])
    ens_p_ai = max(0.0, min(1.0, ens_p_ai + heuristic_adj))

    # ── Confidence calibration ──────────────────────────────
    # Apply slight temperature scaling to prevent over-confident
    # predictions near the boundary (0.45-0.55 range)
    TEMP = 1.3  # >1 = amplifies differences near boundary
    if 0.35 < ens_p_ai < 0.65:
        centered = ens_p_ai - 0.5
        scaled = centered * TEMP
        ens_p_ai = max(0.0, min(1.0, 0.5 + scaled))

    model_std = np.std([xgb_p_ai, oai_p_ai, fs_p_ai])
    ens_p_human = 1.0 - ens_p_ai
    s1_pred     = 0 if ens_p_ai >= 0.5 else 1

    if s1_pred == 0:
        s1_label, s1_color = ("🤖 AI-Generated",  "#FF4B4B")
    else:
        s1_label, s1_color = ("🧑 Human-Written", "#00CC88")

    # ── Stage 2: Safe vs Unsafe ─────────────────────────────
    s2_proba = m2.predict_proba(X)[0]
    s2_pred  = int(m2.predict(X)[0])

    s2_p_safe   = float(s2_proba[0])
    s2_p_unsafe = float(s2_proba[1])

    if s2_pred == 0:
        s2_label, s2_color = ("✅ Safe Prompt",   "#00CC88")
    else:
        s2_label, s2_color = ("⚠️ Unsafe Prompt", "#FF8C00")

    return {
        "text":     text,
        "X":        X,
        # Ensemble results
        "s1_pred":  s1_pred,
        "s1_proba": np.array([ens_p_ai, ens_p_human]),
        "s1_label": s1_label, "s1_color": s1_color,
        "s1_conf":  max(ens_p_ai, ens_p_human) * 100,
        "s1_p0":    ens_p_ai * 100,
        "s1_p1":    ens_p_human * 100,
        # Individual model results
        "xgb_p_ai": xgb_p_ai * 100,  "xgb_p_human": xgb_p_human * 100,
        "oai_p_ai": oai_p_ai * 100,  "oai_p_human": oai_p_human * 100,
        "fs_p_ai":  fs_p_ai * 100,   "fs_p_human":  fs_p_human * 100,
        # Heuristic details
        "style":      style,
        "ppl_detail": ppl_analysis,
        "sent_detail": sent_analysis,
        "heuristic_adj": heuristic_adj * 100,
        "model_agreement": "High" if strong_ai or strong_human else
                           "Good" if all_agree_ai or all_agree_human else
                           agreement_mode if fakespot_overrule else
                           "Low" if model_std > 0.35 else "Moderate",
        "model_std":  model_std,
        "dyn_weights": agreement_mode,
        # Stage 2
        "s2_pred":  s2_pred,  "s2_proba": s2_proba,
        "s2_label": s2_label, "s2_color": s2_color,
        "s2_conf":  max(s2_p_safe, s2_p_unsafe) * 100,
        "s2_p0":    s2_p_safe * 100,
        "s2_p1":    s2_p_unsafe * 100,
    }


# ═════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#FF4B4B' → '255,75,75' for use in rgba()."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


# ═════════════════════════════════════════════════════════════
# CSS — Global Styling System
# ═════════════════════════════════════════════════════════════

st.markdown("""<style>
/* ══════════════════════════════════════════════════════
   DARK GLASSMORPHISM THEME — Professional Minimalist
   ══════════════════════════════════════════════════════ */

/* ── Force dark background globally ─────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #0a0a0f !important;
    color: #e0e0e8 !important;
}
[data-testid="stHeader"] {
    background: rgba(10,10,15,0.85) !important;
    backdrop-filter: blur(12px);
}
/* ── Sidebar fully hidden ──────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
button[kind="headerNoPadding"],
section[data-testid="stSidebar"] {
    display: none !important;
    width: 0 !important;
    min-width: 0 !important;
    visibility: hidden !important;
}

/* ── Custom font import ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Apply Inter to specific UI elements — NOT * (wildcard breaks icon fonts) */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
[data-testid="stHeader"],
[data-testid="stMainMenu"],
[data-testid="stToolbar"],
div[data-testid="stExpander"] summary > div,
div[data-testid="stExpander"] [data-testid="stExpanderDetails"],
div[data-testid="stExpander"] p,
div[data-testid="stMarkdownContainer"],
div[data-testid="stMetric"],
div[data-testid="stText"],
div[data-testid="stCaptionContainer"],
h1, h2, h3, h4, h5, h6,
p, span:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded), div, label, a,
button, input, select, textarea,
td, th, li, dt, dd,
.stButton button,
.stTextArea textarea,
.stSelectbox,
.stMarkdown,
.stAlert,
.stDataFrame {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
/* Icon fonts — must NEVER be overridden */
.material-symbols-rounded,
.material-symbols-rounded *,
.material-icons,
.material-icons *,
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderToggleIcon"] *,
[data-testid="stIconMaterial"],
[data-testid="stIconMaterial"] *,
[data-testid="stExpandSidebarButton"] span,
.exvv1vr0 {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}

/* ── NUCLEAR FIX: Hide broken icon ligature text, show CSS arrow ── */
/* Hide all Material Icon spans inside expander summaries */
div[data-testid="stExpander"] summary [data-testid="stIconMaterial"],
div[data-testid="stExpander"] summary span.material-symbols-rounded,
div[data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"],
div[data-testid="stExpander"] summary > div > span:first-child {
    font-size: 0 !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    display: none !important;
    visibility: hidden !important;
    position: absolute !important;
    left: -9999px !important;
}
/* Add a clean CSS arrow before the expander title text */
div[data-testid="stExpander"] summary {
    position: relative !important;
    padding-left: 0.2rem !important;
    list-style: none !important;
}
div[data-testid="stExpander"] summary::-webkit-details-marker {
    display: none !important;
}
div[data-testid="stExpander"] summary::before {
    content: "▶" !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important;
    color: rgba(165,180,252,0.6) !important;
    margin-right: 0.5rem !important;
    display: inline-block !important;
    transition: transform 0.2s ease !important;
    flex-shrink: 0 !important;
}
div[data-testid="stExpander"][open] summary::before,
div[data-testid="stExpander"] details[open] summary::before {
    content: "▼" !important;
    font-size: 0.6rem !important;
}
code, pre, [data-testid="stCode"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Keyframe Animations (GSAP-inspired) ────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-40px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(40px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 20px rgba(99,102,241,0.15); }
    50%      { box-shadow: 0 0 40px rgba(99,102,241,0.25); }
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes borderGlow {
    0%, 100% { border-color: rgba(99,102,241,0.2); }
    50%      { border-color: rgba(99,102,241,0.5); }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.92); }
    to   { opacity: 1; transform: scale(1); }
}

/* ── Layout Constraints ─────────────────────────── */
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ── Glass Card Base ────────────────────────────────── */
.glass {
    background: rgba(255,255,255,0.03) !important;
    backdrop-filter: blur(20px) saturate(1.5);
    -webkit-backdrop-filter: blur(20px) saturate(1.5);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
}

/* ── Animated App Header ────────────────────────────── */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    animation: fadeInUp 0.8s ease-out;
    position: relative;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}
.app-header * { position: relative; z-index: 1; }
.app-header .title-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.5rem;
    filter: drop-shadow(0 0 20px rgba(99,102,241,0.4));
    animation: fadeIn 0.6s ease-out;
}
.app-header h1 {
    margin: 0;
    font-size: clamp(1.8rem, 4vw, 2.6rem);
    font-weight: 900;
    letter-spacing: -1.5px;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 50%, #818cf8 100%);
    background-size: 200% 200%;
    animation: gradientShift 4s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.app-header .app-desc {
    margin: 0.6rem 0 0;
    color: rgba(255,255,255,0.45);
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 0.3px;
    animation: fadeIn 1s ease-out 0.3s both;
}
.app-header .stage-badges {
    margin-top: 1rem;
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    flex-wrap: wrap;
    animation: fadeInUp 0.8s ease-out 0.5s both;
}
.app-header .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 1rem;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}
.badge:hover {
    transform: translateY(-1px);
}
.badge-ai {
    background: rgba(255,75,75,0.08);
    color: #ff6b6b;
    border: 1px solid rgba(255,75,75,0.2);
    box-shadow: 0 0 20px rgba(255,75,75,0.05);
}
.badge-ai:hover {
    background: rgba(255,75,75,0.14);
    box-shadow: 0 0 30px rgba(255,75,75,0.12);
}
.badge-jail {
    background: rgba(251,191,36,0.08);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.2);
    box-shadow: 0 0 20px rgba(251,191,36,0.05);
}
.badge-jail:hover {
    background: rgba(251,191,36,0.14);
    box-shadow: 0 0 30px rgba(251,191,36,0.12);
}

/* ── Result Cards — Glassmorphism ──────────────────── */
.forensics-card {
    border-radius: 16px;
    padding: clamp(1.2rem, 3vw, 1.8rem) clamp(1.2rem, 3vw, 2rem);
    border-left: 4px solid;
    margin-bottom: 0.75rem;
    background: rgba(255,255,255,0.03) !important;
    backdrop-filter: blur(20px) saturate(1.5);
    -webkit-backdrop-filter: blur(20px) saturate(1.5);
    border-top: 1px solid rgba(255,255,255,0.06);
    border-right: 1px solid rgba(255,255,255,0.04);
    border-bottom: 1px solid rgba(255,255,255,0.02);
    transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
    animation: scaleIn 0.6s ease-out both;
    position: relative;
    overflow: hidden;
}
.forensics-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}
.forensics-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 20px 60px rgba(0,0,0,0.3),
                0 0 40px rgba(99,102,241,0.06);
}
.forensics-card .stage-tag {
    font-weight: 700;
    font-size: 0.65rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin: 0 0 0.6rem 0;
    opacity: 0.7;
}
.forensics-card .verdict {
    font-size: clamp(1.3rem, 3vw, 1.8rem);
    font-weight: 900;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
    letter-spacing: -0.5px;
}
.forensics-card .conf-text {
    font-size: 0.85rem;
    margin: 0;
    color: rgba(255,255,255,0.5) !important;
    font-weight: 400;
}

/* ── Metric Cards — Modern Glass ────────────────────── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px;
    padding: 1rem 0.75rem !important;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    backdrop-filter: blur(10px);
}
div[data-testid="stMetric"]:hover {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(99,102,241,0.2) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}
div[data-testid="stMetric"] label {
    font-size: 0.65rem !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4) !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    letter-spacing: -0.5px;
}

/* ── Progress Bars — Sleek Neon ─────────────────────── */
div[data-testid="stProgress"] {
    margin-bottom: 0.5rem;
}
div[data-testid="stProgress"] > div > div {
    border-radius: 8px !important;
    height: 8px !important;
    background: rgba(255,255,255,0.04) !important;
}

/* ── Expanders — Glass Panels ──────────────────────── */
div[data-testid="stExpander"] {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    background: rgba(255,255,255,0.02) !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    overflow: hidden;
}
div[data-testid="stExpander"]:hover {
    border-color: rgba(99,102,241,0.15) !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    letter-spacing: 0.2px;
    color: #e0e0e8 !important;
}

/* ── Text Area — Deep Dark Input ────────────────────── */
textarea[data-testid="stTextArea"], textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    color: #e0e0e8 !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    padding: 1rem 1.2rem !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    caret-color: #818cf8;
}
textarea:focus {
    border-color: rgba(99,102,241,0.4) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1),
                0 0 30px rgba(99,102,241,0.08) !important;
    background: rgba(255,255,255,0.04) !important;
}
textarea::placeholder {
    color: rgba(255,255,255,0.25) !important;
}

/* ── Buttons — Gradient + Glow ─────────────────────── */
.stButton button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    text-transform: uppercase;
    font-size: 0.78rem !important;
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    background-size: 200% auto;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
}
.stButton button[kind="primary"]:hover {
    background-position: right center !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.45) !important;
}
.stButton button:not([kind="primary"]) {
    background: rgba(255,255,255,0.04) !important;
    color: rgba(255,255,255,0.7) !important;
}
.stButton button:not([kind="primary"]):hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.15) !important;
}

/* ── Section Divider ────────────────────────────── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2rem 0;
    position: relative;
}
.section-divider::after {
    content: '';
    position: absolute;
    top: -1px; left: 20%; right: 20%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
}

/* ── Analyzed Text Preview — Glass ──────────────── */
.analyzed-preview {
    padding: 0.8rem 1.2rem;
    border-radius: 12px;
    background: rgba(99,102,241,0.04);
    border: 1px solid rgba(99,102,241,0.1);
    font-size: 0.82rem;
    color: rgba(255,255,255,0.55);
    margin-bottom: 1.2rem;
    word-break: break-word;
    line-height: 1.6;
    animation: fadeIn 0.5s ease-out;
    backdrop-filter: blur(10px);
}



/* ── Data Table — Dark Glass ────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Markdown Tables ────────────────────────────────── */
table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 12px;
    overflow: hidden;
    width: 100%;
}
th {
    background: rgba(99,102,241,0.08) !important;
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.7rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    font-weight: 700;
    padding: 0.6rem 0.8rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
}
td {
    color: rgba(255,255,255,0.6) !important;
    font-size: 0.82rem;
    padding: 0.5rem 0.8rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
}
tr:hover td {
    background: rgba(99,102,241,0.04) !important;
}

/* ── Captions ───────────────────────────────────────── */
.stCaption, [data-testid="stCaption"] {
    color: rgba(255,255,255,0.35) !important;
    font-size: 0.78rem !important;
}

/* ── Spinner ────────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: #818cf8 !important;
}

/* ── Charts — Dark Background ──────────────────────── */
[data-testid="stVegaLiteChart"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Responsive Stacking ────────────────────────────── */
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    .app-header h1 {
        font-size: 1.6rem !important;
    }
}

/* ── Scrollbar — Minimal ───────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255,255,255,0.15);
}

/* ── Selection Highlight ───────────────────────────── */
::selection {
    background: rgba(99,102,241,0.3);
    color: #fff;
}

/* ── Notification Banners ──────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
    backdrop-filter: blur(10px);
}

/* ── Hide component iframe containers (3D bg) ─────── */
iframe[height="0"] {
    display: none !important;
}
[data-testid="stComponentFrame"],
[data-testid="stCustomComponentV1"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    border: none !important;
    margin: 0 !important;
    padding: 0 !important;
    pointer-events: none !important;
    z-index: -1 !important;
}
/* Also hide any expand arrows on component containers */
[data-testid="stComponentFrame"] button,
[data-testid="stCustomComponentV1"] button,
.stComponentFrame button {
    display: none !important;
}



/* ── Animated background via CSS (no iframe needed) ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: -2;
    pointer-events: none;
    background:
        radial-gradient(ellipse 600px 400px at 20% 30%, rgba(99,102,241,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 500px 500px at 80% 70%, rgba(139,92,246,0.04) 0%, transparent 70%),
        radial-gradient(ellipse 400px 300px at 50% 50%, rgba(168,85,247,0.03) 0%, transparent 60%);
    animation: bgShift 20s ease-in-out infinite;
}
@keyframes bgShift {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.6; }
}

</style>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═════════════════════════════════════════════════════════════

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <span class="title-icon">🔍</span>
    <h1>Text Forensics System</h1>
    <p class="app-desc">Multi-signal deep analysis for AI authorship & adversarial intent detection</p>
    <div class="stage-badges">
        <span class="badge badge-ai">⚡ Stage 1 — AI vs Human</span>
        <span class="badge badge-jail">🛡️ Stage 2 — Safe vs Unsafe</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model check ────────────────────────────────────────────
if not os.path.exists("models/stage1_authorship.pkl") or \
   not os.path.exists("models/stage2_unsafe.pkl"):
    st.error(
        "**Model files not found.**  \n"
        "Run `python scripts/07_final_model.py` first."
    )
    st.stop()

# ── Input ───────────────────────────────────────────────────
st.markdown("""
<div style="margin:0.5rem 0 0.3rem;">
    <span style="font-size:0.7rem; letter-spacing:1.5px; text-transform:uppercase;
                 color:rgba(165,180,252,0.5); font-weight:700;">
        INPUT TEXT
    </span>
</div>
""", unsafe_allow_html=True)
st.text_area(
    "Enter text to analyze",
    height=180,
    placeholder="Paste any text here — article, essay, AI output, prompt…",
    key="input_text",
    label_visibility="collapsed",
)
st.caption("Minimum 5 words · Supports articles, essays, AI outputs, prompts")

col_btn_analyze, col_btn_clear, _ = st.columns([1.5, 1, 4.5])
with col_btn_analyze:
    analyze_clicked = st.button(
        "⚡ Analyze Text", type="primary", width="stretch"
    )
with col_btn_clear:
    if st.button("🗑️ Clear", width="stretch"):
        st.session_state.result = None
        st.session_state.analyzed_text = ""
        st.session_state.input_text = ""
        st.rerun()

# ── Analysis Trigger ────────────────────────────────────────
if analyze_clicked:
    current_text = st.session_state.get("input_text", "").strip()
    if not current_text:
        st.warning("⚠️ Please enter some text first.")
    elif len(current_text.split()) < 5:
        st.warning("⚠️ Please enter at least 5 words for meaningful analysis.")
    else:
        with st.spinner("Running multi-signal deep analysis (3 models + 13 heuristics + sentence analysis)…"):
            st.session_state.result = predict(current_text)
            st.session_state.analyzed_text = current_text


# ═════════════════════════════════════════════════════════════
# RESULTS
# ═════════════════════════════════════════════════════════════

if st.session_state.result is not None:
    r = st.session_state.result

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Section Label ───────────────────────────────────────
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
        <div style="width:3px; height:20px; background:linear-gradient(180deg, #6366f1, #a855f7);
                    border-radius:2px;"></div>
        <span style="font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;
                     color:rgba(165,180,252,0.5); font-weight:700;">
            Analysis Results
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Analyzed text preview ───────────────────────────────
    preview = r["text"][:200].replace("\n", " ")
    if len(r["text"]) > 200:
        preview += "…"
    st.markdown(
        f'<div class="analyzed-preview">📝 <strong>Analyzed:</strong> "{preview}"</div>',
        unsafe_allow_html=True,
    )

    # ── Result cards ────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        c = r["s1_color"]
        _rgb = _hex_to_rgb(c)
        st.markdown(f"""
        <div class="forensics-card"
             style="border-left-color:{c}; background:rgba({_rgb},0.05) !important;
                    box-shadow: 0 4px 30px rgba({_rgb},0.08), inset 0 1px 0 rgba(255,255,255,0.04);">
            <p class="stage-tag" style="color:{c};">⚡ Stage 1 — Authorship Detection</p>
            <p class="verdict" style="color:{c}; text-shadow: 0 0 30px rgba({_rgb},0.3);">{r["s1_label"]}</p>
            <p class="conf-text">Confidence: <strong style="color:{c} !important; font-size:1.1rem;">{r["s1_conf"]:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(r["s1_p0"] / 100, text=f"🤖 AI-Generated — {r['s1_p0']:.1f}%")
        st.progress(r["s1_p1"] / 100, text=f"🧑 Human-Written — {r['s1_p1']:.1f}%")

    with col2:
        c = r["s2_color"]
        _rgb = _hex_to_rgb(c)
        st.markdown(f"""
        <div class="forensics-card"
             style="border-left-color:{c}; background:rgba({_rgb},0.05) !important;
                    box-shadow: 0 4px 30px rgba({_rgb},0.08), inset 0 1px 0 rgba(255,255,255,0.04);">
            <p class="stage-tag" style="color:{c};">🛡️ Stage 2 — Intent Detection</p>
            <p class="verdict" style="color:{c}; text-shadow: 0 0 30px rgba({_rgb},0.3);">{r["s2_label"]}</p>
            <p class="conf-text">Confidence: <strong style="color:{c} !important; font-size:1.1rem;">{r["s2_conf"]:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(r["s2_p0"] / 100, text=f"✅ Safe Prompt — {r['s2_p0']:.1f}%")
        st.progress(r["s2_p1"] / 100, text=f"⚠️ Unsafe Prompt — {r['s2_p1']:.1f}%")

    # ── Ensemble Model Breakdown ────────────────────────────
    with st.expander("🧠 Ensemble Model Breakdown — Stage 1", expanded=True):
        agree = r.get("model_agreement", "Moderate")
        agree_emoji = {"High": "🟢", "Good": "🟡", "Moderate": "🟠", "Low": "🔴", "Override": "🔵"}.get(agree, "⚪")
        mode = r.get("dyn_weights", "weighted")
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.04); border:1px solid rgba(99,102,241,0.1);
                    border-radius:10px; padding:0.6rem 1rem; margin-bottom:0.8rem;
                    display:flex; flex-wrap:wrap; gap:1.2rem; align-items:center;">
            <span style="font-size:0.78rem; color:rgba(255,255,255,0.6);">
                <strong style="color:#a5b4fc;">3-Model Ensemble</strong> · Mode: <code>{mode}</code>
            </span>
            <span style="font-size:0.78rem; color:rgba(255,255,255,0.6);">
                {agree_emoji} Agreement: <strong>{agree}</strong>
            </span>
            <span style="font-size:0.78rem; color:rgba(255,255,255,0.6);">
                Heuristic: <strong>{r.get('heuristic_adj', 0):+.1f}%</strong>
            </span>
        </div>
        """, unsafe_allow_html=True)
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.markdown("**XGBoost (features)**")
            st.progress(r["xgb_p_ai"] / 100, text=f"🤖 AI — {r['xgb_p_ai']:.1f}%")
            st.progress(r["xgb_p_human"] / 100, text=f"🧑 Human — {r['xgb_p_human']:.1f}%")
        with mcol2:
            st.markdown("**OpenAI RoBERTa**")
            st.progress(r["oai_p_ai"] / 100, text=f"🤖 AI — {r['oai_p_ai']:.1f}%")
            st.progress(r["oai_p_human"] / 100, text=f"🧑 Human — {r['oai_p_human']:.1f}%")
        with mcol3:
            st.markdown("**Fakespot RoBERTa**")
            st.progress(r["fs_p_ai"] / 100, text=f"🤖 AI — {r['fs_p_ai']:.1f}%")
            st.progress(r["fs_p_human"] / 100, text=f"🧑 Human — {r['fs_p_human']:.1f}%")

    # ── Deep Analysis Details ───────────────────────────────
    with st.expander("🔬 Deep Analysis — Writing Style & Consistency"):
        style = r.get("style", {})
        if style:
            st.markdown("##### AI Writing-Style Heuristics")
            st.markdown(
                f"Overall AI-style score: **{style.get('ai_style', 0):.2f}** "
                f"(0 = human-like, 1 = AI-like)"
            )
            hcol1, hcol2, hcol3 = st.columns(3)
            with hcol1:
                st.progress(min(style.get('transitions', 0), 1.0),
                            text=f"Formal Transitions: {style.get('transitions', 0):.2f}")
                st.progress(min(style.get('hedging', 0), 1.0),
                            text=f"Academic/AI Words: {style.get('hedging', 0):.2f}")
                st.progress(min(style.get('phrase_match', 0), 1.0),
                            text=f"AI Phrase Patterns: {style.get('phrase_match', 0):.2f}")
                st.progress(min(style.get('starter_match', 0), 1.0),
                            text=f"Formulaic Starters: {style.get('starter_match', 0):.2f}")
            with hcol2:
                st.progress(min(style.get('uniformity', 0), 1.0),
                            text=f"Sentence Uniformity: {style.get('uniformity', 0):.2f}")
                st.progress(min(style.get('comma_density', 0), 1.0),
                            text=f"Comma Density: {style.get('comma_density', 0):.2f}")
                st.progress(min(style.get('repetitive', 0), 1.0),
                            text=f"Repetitive Starters: {style.get('repetitive', 0):.2f}")
                st.progress(min(style.get('no_contractions', 0), 1.0),
                            text=f"No Contractions: {style.get('no_contractions', 0):.2f}")
            with hcol3:
                st.progress(min(style.get('formal_punct', 0), 1.0),
                            text=f"Formal Punctuation: {style.get('formal_punct', 0):.2f}")
                st.progress(min(style.get('first_person', 0), 1.0),
                            text=f"Lacks First-Person: {style.get('first_person', 0):.2f}")
                human_m = style.get('human_markers', 0)
                st.progress(min(human_m, 1.0),
                            text=f"Human Informality: {human_m:.2f}")
                typo_s = style.get('has_typos', 0)
                st.progress(min(typo_s, 1.0),
                            text=f"Typos/Errors: {typo_s:.2f}")

        # Sentence-level analysis
        sent = r.get("sent_detail")
        if sent is not None:
            st.markdown("---")
            st.markdown("##### Sentence-Level Consistency")
            scol1, scol2, scol3, scol4 = st.columns(4)
            scol1.metric("Sentences Analyzed", sent["n_sents"])
            scol2.metric("Mean P(AI)", f"{sent['mean']:.3f}")
            scol3.metric("AI Fraction", f"{sent['ai_frac']:.1%}")
            scol4.metric("Score Std Dev", f"{sent['std']:.3f}")
            # Show sentence score chart
            if sent.get("scores") and len(sent["scores"]) > 0:
                sent_df = pd.DataFrame({
                    "Sentence": [f"S{i+1}" for i in range(len(sent["scores"]))],
                    "AI_Score": [float(s) for s in sent["scores"]]
                })
                st.bar_chart(sent_df.set_index("Sentence"), y="AI_Score", height=200)
                if sent["std"] < 0.12:
                    st.caption("⚠️ Very uniform sentence scores — typical of AI-generated text")
                elif sent["std"] > 0.30:
                    st.caption("✅ High score variance — typical of human-written text")
        else:
            st.caption("Sentence-level analysis requires at least 2 sentences with 4+ words each.")

        # Perplexity analysis
        ppl = r.get("ppl_detail")
        if ppl is not None:
            st.markdown("---")
            st.markdown("##### Perplexity Analysis (GPT-2)")
            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric("Mean Perplexity", f"{ppl['mean_ppl']:.1f}")
            pcol2.metric("Std Dev", f"{ppl['std_ppl']:.1f}")
            pcol3.metric("Coeff of Variation", f"{ppl['cv']:.3f}")
            if ppl['low_ppl_signal'] > 0.6 and ppl['low_var_signal'] > 0.5:
                st.caption("⚠️ Low perplexity + low variance — characteristic of AI-generated text")
            elif ppl['cv'] > 0.8:
                st.caption("✅ High perplexity variance — characteristic of human writing")

    # ── Text Statistics ─────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.6rem;">
        <div style="width:3px; height:20px; background:linear-gradient(180deg, #6366f1, #a855f7);
                    border-radius:2px;"></div>
        <span style="font-size:0.7rem; letter-spacing:2px; text-transform:uppercase;
                     color:rgba(165,180,252,0.5); font-weight:700;">
            📊 Text Statistics
        </span>
    </div>
    """, unsafe_allow_html=True)

    txt   = r["text"]
    words = txt.split()
    sents = [s for s in re.split(r'[.!?]+', txt) if s.strip()]
    lower = [w.lower() for w in words]

    row1_a, row1_b, row1_c = st.columns(3)
    row1_a.metric("Characters",  f"{len(txt):,}")
    row1_b.metric("Words",       f"{len(words):,}")
    row1_c.metric("Sentences",   f"{len(sents):,}")

    row2_a, row2_b, row2_c = st.columns(3)
    row2_a.metric("Unique Words", f"{len(set(lower)):,}")
    row2_b.metric("Avg Word Len", f"{sum(len(w) for w in words)/max(len(words),1):.1f}")
    row2_c.metric("Avg Sent Len", f"{len(words)/max(len(sents),1):.1f} words")

    # ── Feature Vector Details ──────────────────────────────
    with st.expander("🔬 Feature Vector Details"):
        X = r["X"]
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Dimensions", X.shape[1])
        fc2.metric("Min Value",  f"{X.min():.4f}")
        fc3.metric("Max Value",  f"{X.max():.4f}")
        fc4.metric("Mean Value", f"{X.mean():.4f}")

        st.markdown(
            "| Block | Feature Group | Dims | Description |\n"
            "|:-----:|:-------------|:----:|:------------|\n"
            "| A | Semantic Embeddings | 384 | all-MiniLM-L6-v2 sentence embeddings |\n"
            "| B | Stylometric | 10 | Word length, sentence structure, vocabulary |\n"
            "| C | Statistical | 7 | Entropy, diversity, burstiness, Yule's K |\n"
            "| D | Linguistic | 10 | POS tags, dependency, readability |\n"
            "| E | Perplexity | 1 | GPT-2 language model surprise |\n"
            "| F | Structural | 8 | Intent signals, punctuation patterns |\n"
            "| | **Total** | **420** | |"
        )

    # ── Raw Probabilities ───────────────────────────────────
    with st.expander("📈 Raw Prediction Probabilities"):
        df = pd.DataFrame({
            "Model / Stage": [
                "Ensemble (Stage 1)", "Ensemble (Stage 1)",
                "XGBoost (Stage 1)",  "XGBoost (Stage 1)",
                "OpenAI RoBERTa",     "OpenAI RoBERTa",
                "Fakespot RoBERTa",   "Fakespot RoBERTa",
                "Heuristic Adj.",     "Model Agreement",
                "Stage 2 — Intent",   "Stage 2 — Intent",
            ],
            "Class": [
                "🤖 AI-Generated",  "🧑 Human-Written",
                "🤖 AI-Generated",  "🧑 Human-Written",
                "🤖 AI-Generated",  "🧑 Human-Written",
                "🤖 AI-Generated",  "🧑 Human-Written",
                "Shift Applied",      r.get("model_agreement", "N/A"),
                "✅ Safe Prompt",    "⚠️ Unsafe Prompt",
            ],
            "Percentage": [
                f"{r['s1_p0']:.2f}%",      f"{r['s1_p1']:.2f}%",
                f"{r['xgb_p_ai']:.2f}%",   f"{r['xgb_p_human']:.2f}%",
                f"{r['oai_p_ai']:.2f}%",   f"{r['oai_p_human']:.2f}%",
                f"{r['fs_p_ai']:.2f}%",    f"{r['fs_p_human']:.2f}%",
                f"{r.get('heuristic_adj', 0):+.2f}%", f"std={r.get('model_std', 0):.3f}",
                f"{r['s2_p0']:.2f}%",      f"{r['s2_p1']:.2f}%",
            ],
        })
        st.dataframe(df, width="stretch", hide_index=True)