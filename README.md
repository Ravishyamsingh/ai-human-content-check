# Two-Stage Text Forensics System

A machine learning pipeline for **AI-generated text detection** and **jailbreak prompt classification**, designed for academic research (thesis-grade reproducibility).

## Overview

| Stage | Task | Classes |
|-------|------|---------|
| **Stage 1** | AI vs Human authorship detection | 0 = AI-Generated, 1 = Human-Written |
| **Stage 2** | Jailbreak prompt detection | 0 = Safe Prompt, 1 = Jailbreak Prompt |

Both stages use a **420-dimensional feature vector** extracted from each text:

| Block | Features | Dims |
|-------|----------|------|
| A | Semantic Embeddings (all-MiniLM-L6-v2) | 384 |
| B | Stylometric (word/sentence style) | 10 |
| C | Statistical (entropy, diversity) | 7 |
| D | Linguistic (POS tags, readability via SpaCy) | 10 |
| E | Perplexity (GPT-2) | 1 |
| F | Structural (intent signals) | 8 |
| **Total** | | **420** |

## Project Structure

```
scripts/
  00_install.py            # One-time setup: install packages + download models
  01_download_data.py      # Download datasets from HuggingFace
  02_preprocess.py         # Minimal-intervention text cleaning
  03_eda.py                # Exploratory data analysis with significance tests
  04_feature_extraction.py # Extract 420-dim features (GPU-accelerated)
  05_model_screening.py    # Screen 13 classifiers via 5-fold stratified CV
  06_stability_testing.py  # Test top-5 models across 5 random seeds
  07_final_model.py        # Train final model, save pipeline + holdout set
  08_holdout_eval.py       # Final holdout evaluation (run once)

app/
  streamlit_app.py         # Interactive web UI for inference

data/
  raw/                     # Downloaded CSVs
  processed/               # Cleaned CSVs + feature matrices (.npy)
  features/                # Per-block feature checkpoints
  holdout/                 # Held-out test sets (never seen during training)

models/                    # Saved sklearn pipelines (.pkl)
results/                   # CSVs, plots, reports per stage
run_all.py                 # Run the full pipeline (scripts 02–08)
```

## Setup

```bash
# 1. Install dependencies
python scripts/00_install.py

# 2. Download datasets
python scripts/01_download_data.py
```

## Running the Pipeline

```bash
# Run all steps (preprocessing → EDA → features → screening → stability → final model → holdout)
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

## Interactive App

```bash
streamlit run app/streamlit_app.py
```

Paste any text into the web interface to get:
- **Stage 1**: AI vs Human authorship classification with confidence score
- **Stage 2**: Safe vs Jailbreak prompt classification with confidence score

## Requirements

- Python 3.8+
- GPU recommended (for feature extraction speed), but CPU works
- ~4 GB disk space for models and data

See [requirements.txt](requirements.txt) for full dependency list.

## Methodology Notes

- **Preprocessing**: Minimal-intervention — only fixes encoding artifacts and whitespace. Never touches capitalization, punctuation, or stopwords (these ARE the features).
- **Class imbalance**: Handled via SMOTE on training folds only (never on validation/test).
- **Model selection**: Composite ranking score = 0.6 × F1-Macro + 0.3 × Balanced Accuracy − 0.1 × F1-Std.
- **Stability**: Top-5 models tested across 5 random seeds to confirm results are not seed-dependent.
- **Holdout**: 10% of data held out and never touched until final evaluation.



Conduct a comprehensive review of the entire project and identify the root cause of the issue. I have already implemented certain fixes, yet the system continues to classify AI-generated content as human-written prompts.
Perform a detailed analysis to clearly determine:
Which parts of the content are AI-generated
Which parts are genuinely human-written
Why the detection mechanism is failing
Additionally:
Review and improve the UI/UX to ensure it follows a modern design approach.
Ensure all content is properly visible and responsive on desktop view.
Verify which model is being used for AI-to-human content detection or transformation.
Confirm whether the selected model is functioning correctly and configured properly.
Identify any logical, architectural, or implementation-level issues affecting performance.
Provide a thorough technical assessment and accurate conclusions based on deep analysis. Don't write any content in chat and don't make  .MD file, Only focus on work



Loading weights:  97%|#########7| 144/148 [00:00<00:00, 733.20it/s, Materializing param=transformer.h.11.mlp.c_proj.weight]
Loading weights:  98%|#########7| 145/148 [00:00<00:00, 733.20it/s, Materializing param=transformer.ln_f.bias]             
Loading weights:  98%|#########7| 145/148 [00:00<00:00, 733.20it/s, Materializing param=transformer.ln_f.bias]
Loading weights:  99%|#########8| 146/148 [00:00<00:00, 733.20it/s, Materializing param=transformer.ln_f.weight]
Loading weights:  99%|#########8| 146/148 [00:00<00:00, 733.20it/s, Materializing param=transformer.ln_f.weight]
Loading weights:  99%|#########9| 147/148 [00:00<00:00, 733.20it/s, Materializing param=transformer.wpe.weight] 
Loading weights:  99%|#########9| 147/148 [00:00<00:00, 733.20it/s, Materializing param=transformer.wpe.weight]
Loading weights: 100%|##########| 148/148 [00:00<00:00, 733.20it/s, Materializing param=transformer.wte.weight]
Loading weights: 100%|##########| 148/148 [00:00<00:00, 733.20it/s, Materializing param=transformer.wte.weight]
Loading weights: 100%|##########| 148/148 [00:00<00:00, 715.80it/s, Materializing param=transformer.wte.weight]
`loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.


cd "D:\ai-human content"; .\venv\Scripts\Activate.ps1; streamlit run app/streamlit_app.py 2>&1