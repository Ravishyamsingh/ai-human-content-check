# AI vs Human Content Detection System

A two-stage ML system that detects **AI-generated text** and classifies **unsafe prompts** — powered by a 3-model ensemble (XGBoost + two RoBERTa transformers) with 13 writing-style heuristics.

> For a detailed technical deep-dive, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)

---

## Quick Setup

### Prerequisites

- **Python 3.10+** (tested on 3.12)
- **Git**
- ~4 GB disk space (for models and data)
- GPU recommended but not required

### 1. Clone & Create Virtual Environment

```bash
git clone https://github.com/Ravishyamsingh/ai-human-content-check.git
cd ai-human-content-check

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLP Models

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords')"
```

### 4. Download Datasets (only if retraining)

```bash
python scripts/01_download_data.py
```

---

## Running the App

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`. Paste any text to get:
- **Stage 1** — AI vs Human authorship detection with confidence %
- **Stage 2** — Unsafe prompt classification

---

## Retraining the Pipeline (Optional)

If you want to retrain from scratch:

```bash
# Run all steps: preprocess → EDA → features → screening → stability → final model → holdout
python run_all.py

# Or run individual steps:
python scripts/02_preprocess.py
python scripts/03_eda.py
python scripts/04_feature_extraction.py
python scripts/05_model_screening.py
python scripts/06_stability_testing.py
python scripts/07_final_model.py
python scripts/08_holdout_eval.py
```

---

## Project Structure

```
app/streamlit_app.py       → Web UI + inference engine (3-model ensemble)
scripts/00_install.py      → One-time package installer
scripts/01_download_data.py→ Download datasets from HuggingFace
scripts/02_preprocess.py   → Text cleaning (minimal-intervention)
scripts/03_eda.py          → Exploratory data analysis
scripts/04_feature_extraction.py → Extract 420-dim feature vectors
scripts/05_model_screening.py    → Screen 13 classifiers (5-fold CV)
scripts/06_stability_testing.py  → Test top-5 across 5 random seeds
scripts/07_final_model.py        → Train & save final model
scripts/08_holdout_eval.py       → Final holdout evaluation
data/                      → Raw CSVs, processed data, feature matrices
models/                    → Saved model pipelines (.pkl)
results/                   → Reports, plots, screening results
run_all.py                 → Runs the full pipeline (scripts 02–08)
```

---

## License

This project is for academic/research purposes.