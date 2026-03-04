"""
Run this ONCE to set up everything.
"""
import subprocess
import sys

packages = [
    "datasets",
    "pandas",
    "numpy",
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "matplotlib",
    "seaborn",
    "spacy",
    "transformers",
    "torch",
    "sentence-transformers",
    "streamlit",
    "joblib",
    "scipy",
    "tqdm"
]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Download spacy model
subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Download NLTK stopwords (needed by 04_feature_extraction.py)
import nltk
nltk.download("stopwords", quiet=True)

print("✅ All packages and data installed!")