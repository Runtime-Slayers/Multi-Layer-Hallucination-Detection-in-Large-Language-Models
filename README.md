# Multi-Layer AI Hallucination Detector for Academic Writing

[![CI Status](https://github.com/Runtime-Slayers/Multi-Layer-Hallucination-Detection-in-Large-Language-Models/workflows/Python%20CI/badge.svg)](https://github.com/Runtime-Slayers/Multi-Layer-Hallucination-Detection-in-Large-Language-Models/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)

An automated multi-layer forensic detection framework for identifying large language model (LLM) hallucinations in academic publications. Built on three independent layers—**Citation Forensics**, **Statistical Forensics**, and **Confidence-Accuracy Mismatch Detection**—integrated using machine learning classifiers.

---

## 1. Mathematical Foundations

The detector uses three distinct layers to compute suspiciousness scores across text metrics:

1.  **Citation forensics suspicion score** $S_{\text{cite}} \in [0, 1]$:
    $$S_{\text{cite}} = \max\left(0, \min\left(1, 0.5 \cdot F_{\text{fake}} + 0.3 \cdot F_{\text{future}} + 0.1 \cdot (1 - F_{\text{real}}) + 0.1 \cdot \frac{N_{\text{ghost}}}{N_{\text{total}}}\right)\right)$$
2.  **Statistical forensics suspicion score** $S_{\text{stat}} \in [0, 1]$:
    $$S_{\text{stat}} = 0.6 \cdot \frac{N_{\text{suspicious}}}{N_{\text{total\_stats}}} + 0.4 \cdot F_{\text{round\_percent}}$$
3.  **Confidence-accuracy mismatch score** $S_{\text{conf}} \in [0, 1]$:
    $$S_{\text{conf}} = \max\left(0, \min\left(1, 0.6 \cdot F_{\text{high\_conf}} + 0.4 \cdot \frac{N_{\text{conf\_phrases}}}{10}\right)\right)$$

For detailed derivations and definitions of individual terms, see the [Theory Documentation](docs/THEORY.md).

---

## 2. Package Architecture

```
Multi-Layer-Hallucination-Detection-in-Large-Language-Models/
├── hallucination_detector/         # Core python package
│   ├── __init__.py                # Package exports
│   ├── corpus.py                  # Synthetic academic paper & claim generator
│   ├── citation.py                # Citation forensics module
│   ├── statistics.py              # Statistical heuristics forensics module
│   ├── confidence.py              # Linguistic overconfidence analyzer
│   ├── classifier.py              # Cross-validated Random Forest, LR, GB models
│   └── utils.py                   # Plotting & results serialization utilities
├── experiments/                   # Experiment scripts
│   └── p9_hallucination_validation.py  # Parameterized runner
├── docs/                          # Documentation
│   └── THEORY.md                  # Comprehensive mathematical details
├── tests/                         # Unit tests
│   └── test_detector.py
├── pyproject.toml                 # Standard packaging config
└── requirements.txt               # Dependencies
```

---

## 3. Installation

Install the package in editable mode along with development dependencies:

```bash
pip install -e .[dev]
```

---

## 4. Usage

### Command Line Interface

Run the validation experiment runner via the automated script entrypoint:

```bash
# Run with default settings (600 papers, 45% hallucination rate)
hallucination-validate

# Configure parameters and skip plotting
hallucination-validate --n-papers 1000 --hallucination-rate 0.35 --no-plots
```

### Python API

```python
from hallucination_detector import generate_corpus, train_and_evaluate

# 1. Generate synthetic corpus
papers = generate_corpus(n_papers=200, hallucination_rate=0.45)

# 2. Extract features and train classifiers
results = train_and_evaluate(papers, seed=42)

print(f"Random Forest Test AUC: {results['final_auc']:.4f}")
```

---

## 5. Performance Results

A Random Forest classifier evaluated using 5-fold cross-validation with added Gaussian measurement noise ($\sigma = 10\%$):

| Model | 5-Fold CV ROC-AUC | Out-of-Fold Std Dev |
| :--- | :---: | :---: |
| Logistic Regression | 0.923 | 0.014 |
| **Random Forest (100 Trees)** | **0.909** | **0.013** |
| Gradient Boosting | 0.904 | 0.008 |

---

## 6. License

This codebase is licensed under the **MIT License**. The text, report, and paper are licensed under **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**.

© 2026 Runtime-Slayers / Bhavanam Rajendra Reddy et al. All rights reserved.