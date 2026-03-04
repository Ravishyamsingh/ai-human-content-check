import numpy as np
import pandas as pd
import re
import string
import spacy
from collections import Counter
import nltk
from nltk.corpus import stopwords
import joblib
import torch
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# 1. Setup and loading
print("--- [1/6] Loading libraries and NLTK data... ---")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm', disable=['ner'])

# 2. Load Data
print("--- [2/6] Loading data files... ---")
try:
    df = pd.read_csv('data/processed/stage1_clean.csv')
    text = df['text'].iloc[0]
    X_saved = np.load('data/processed/stage1_features.npy')
    print(f"    Loaded text (len: {len(text)}) and saved features.")
except FileNotFoundError as e:
    print(f"!!! Error: Could not find file. {e}")
    exit()

# 3. Load Models (This is the slow part)
print("--- [3/6] Loading AI Models (this may take a minute)... ---")
embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
tok = GPT2TokenizerFast.from_pretrained('gpt2')
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2.eval()

# 4. Feature Extraction - Manual Math
print("--- [4/6] Calculating Manual Features (Block B, F)... ---")
words = text.split()
sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
wc = Counter(words)

# Block B
B = np.array([
    np.mean([len(w) for w in words]), 
    len(words)/max(len(sentences),1),
    np.std([len(s.split()) for s in sentences]) if len(sentences)>1 else 0,
    len(set(words))/len(words), 
    sum(1 for c in wc.values() if c==1)/len(words),
    sum(1 for w in words if w.lower() in stop_words)/len(words),
    sum(1 for c in text if c in string.punctuation)/len(text),
    sum(1 for c in text if c.isupper())/len(text),
    sum(1 for c in text if c.isdigit())/len(text),
    sum(1 for c in text if not c.isalnum() and c not in string.whitespace)/len(text)
])

# Block D Preparation
doc = nlp(text[:5000])
tokens = [t for t in doc if not t.is_space]
pos_c = Counter(t.pos_ for t in tokens)
total = len(tokens) if len(tokens) > 0 else 1
nw = max(len(words),1)
ns = max(len(sentences),1)

def syl(w):
    w=w.lower(); c=0; pv=False
    for ch in w:
        v=ch in 'aeiouy'
        if v and not pv: c+=1
        pv=v
    if w.endswith('e') and c>1: c-=1
    return max(1,c)

sc = sum(syl(w) for w in words)
fk = 206.835 - 1.015*(nw/ns) - 84.6*(sc/nw)

# Block D
D = np.array([
    pos_c.get('NOUN',0)/total, pos_c.get('VERB',0)/total,
    pos_c.get('ADJ',0)/total, pos_c.get('ADV',0)/total, pos_c.get('PRON',0)/total,
    (pos_c.get('CCONJ',0)+pos_c.get('SCONJ',0))/total,
    len(pos_c)/17.0, np.mean([len(list(t.children)) for t in doc]) if len(doc)>0 else 0,
    sum(1 for w in words if w and w[0].isupper())/nw,
    max(0.0,min(100.0,fk))/100.0
])

# Block F
F = np.array([
    text.count('?')/len(text), 
    text.count('!')/len(text),
    sum(1 for s in sentences if s.split() and s.split()[0].lower() in {'ignore','tell','give','write'})/len(sentences) if len(sentences) > 0 else 0,
    text.lower().count('you')/len(words), 
    text.lower().count('can')/len(words),
    text.count('\"')/len(text),  # Note: In a file, '\"' works fine. In CLI, this breaks.
    sum(1 for w in words if w.isupper())/len(words),
    np.var([len(s.split()) for s in sentences]) if len(sentences)>1 else 0
])

# 5. Feature Extraction - AI Models
print("--- [5/6] Calculating Embeddings and Perplexity... ---")
semantic = embed.encode([text], batch_size=1, show_progress_bar=False)
enc = tok(text[:1000], return_tensors='pt', truncation=True)
with torch.no_grad():
    ppl = np.exp(gpt2(**enc, labels=enc['input_ids']).loss.item())

# Assemble vector
# Note: Original script had np.zeros(7) to account for missing features in indices 394-400
X_app = np.concatenate([semantic[0], B, np.zeros(7), D, [ppl], F]).reshape(1,-1)

# 6. Inference and Comparison
print("--- [6/6] Running Prediction... ---")
m1 = joblib.load('models/stage1_authorship.pkl')

print('\nRESULTS:')
print('Direct on saved features:', m1.predict(X_saved[0:1])[0], m1.predict_proba(X_saved[0:1])[0].round(3))
print('App reconstructed vector:', m1.predict(X_app)[0], m1.predict_proba(X_app)[0].round(3))
print('-'*30)
print('Block B match:', np.allclose(X_saved[0][384:394], B, atol=1e-3))
print('Block D match:', np.allclose(X_saved[0][401:411], D, atol=1e-3))
print('Block F match:', np.allclose(X_saved[0][412:420], F, atol=1e-3))
print('Block D saved:', X_saved[0][401:411].round(4))
print('Block D app  :', D.round(4))