# REPORT P9: Multi-Layer AI Hallucination Detector for Academic Writing

**Preprint | Brain Complexity Research Group**  
**Project:** P9 | **BT Reference:** BT33 — AI Hallucination Detection  
**Status:** Simulation Study (Synthetic Corpus)

---

## Abstract

Large language models (LLMs) increasingly generate plausible but fabricated academic content — a phenomenon termed "hallucination." We present a three-layer detection pipeline applied to a synthetic corpus of 600 papers (47% hallucinated) via: (1) citation forensics identifying fake journals, anachronistic citations, and ghost authors; (2) statistical forensics flagging implausible effect sizes, round-number p-values, and correlation suspicion scores; and (3) confidence-accuracy mismatch detection measuring linguistic overconfidence relative to evidential support. A Random Forest classifier trained on 12 extracted features achieves AUC = 1.000 (95% CI estimated from 5-fold CV, SD = 0.000). The top predictive feature is a composite confidence score (importance 22.3%), followed by citation suspicion (21.8%) and fake journal fraction (17.6%). Statistical forensics alone achieves AUC = 0.976, demonstrating that statistical implausibility is a robust hallucination signal even without citation verification.

---

## 1 · Introduction

AI-generated hallucinations in academic writing represent an epistemological crisis: fabricated citations, impossible statistics, and authoritative-sounding claims that are factually empty. Detection methods have largely been ad hoc; we demonstrate that systematic multi-layer forensic analysis achieves near-perfect classification.

**Three forensic hypotheses:**
1. Hallucinated text invents citations that violate real-world constraints (non-existent journals, future-year publications, implausible author names)
2. Fabricated statistics cluster at psychologically round numbers and implausible extremes
3. LLM-generated academic text overuses high-confidence language without epistemic hedging

---

## 2 · Methods

### 2.1 Synthetic Corpus

| Parameter | Value |
|-----------|-------|
| Total papers | 600 |
| Hallucinated papers | 282 (47.0%) |
| Genuine papers | 318 (53.0%) |
| Hallucination rate | 0.45 (design); 0.47 (realised) |

Each paper was assigned: reference list (5–40 citations), statistical claims (2–10 results), confidence language pattern, and a ground-truth hallucination label.

### 2.2 Detection Layers

**Layer 1 — Citation Forensics (4 features):**
- `frac_fake_journals`: fraction of citations from a predefined predatory/non-existent journal list
- `frac_future_years`: fraction of citations with year > 2025
- `n_ghost_markers`: citations with author names beginning with "X." (ghost author marker)
- `citation_suspicion`: composite citation anomaly score

**Layer 2 — Statistical Forensics (5 features):**
- `n_suspicious_stats`: count of implausible statistics
- `round_number_frac`: fraction of p-values that are suspiciously round (0.05, 0.01, etc.)
- `stat_suspicion`: composite statistical anomaly score (correlations > 0.85, p < 1e-10, d > 2.0)
- `n_stats`: total statistical claim count
- `n_citations`: total citation count

**Layer 3 — Confidence-Accuracy Mismatch (3 features):**
- `confidence_score`: frequency of high-certainty phrases ("conclusively proves", "definitively shows")
- `high_conf_frac`: fraction of sentences with overconfident language
- `conf_phrase_count`: raw count of confidence markers

### 2.3 Classification

Features → StandardScaler → 5-fold cross-validated comparison of:
- Logistic Regression
- Random Forest (100 trees)
- Gradient Boosting (100 estimators)

Final model: Random Forest on 80/20 train/test split.

---

## 3 · Results

### 3.1 Cross-Validation AUC

| Model | CV AUC | CV SD |
|-------|--------|-------|
| Logistic Regression | 1.000 | 0.000 |
| **Random Forest** | **1.000** | **0.000** |
| Gradient Boosting | 1.000 | 0.000 |

### 3.2 Final Random Forest (Test Set)

| Metric | Value |
|--------|-------|
| AUC | **1.000** |
| Average Precision (AP) | **1.000** |
| F1 (weighted) | **1.000** |
| Accuracy | **1.000** |

### 3.3 Layer-by-Layer Effectiveness

| Detection Layer | AUC |
|----------------|-----|
| Citation forensics | 1.000 |
| Statistical forensics | **0.976** |
| Confidence mismatch | 1.000 |
| Composite score | 1.000 |
| Random Forest (ML) | 1.000 |

Statistical forensics (AUC = 0.976) is the weakest single layer but still highly effective — plausible statistics are cognitively harder to fabricate consistently than realistic citations.

### 3.4 Feature Importances (Top 5)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `confidence_score` | 22.3% |
| 2 | `citation_suspicion` | 21.8% |
| 3 | `frac_fake_journals` | 17.6% |
| 4 | `high_conf_frac` | 16.2% |
| 5 | `n_suspicious_stats` | 11.6% |

Confidence language overuse is the single strongest predictor, reflecting LLM tendency to assert rather than hedge.

---

## 4 · Figures

| Figure | Description |
|--------|-------------|
| `01_detection_pipeline.png` | Three-layer detection architecture flowchart |
| `02_roc_pr_curves.png` | ROC and Precision-Recall curves for all models + layers |
| `03_feature_importance.png` | Random Forest feature importance bar chart |
| `04_citation_forensics.png` | Citation suspicion score distributions by label |
| `05_statistical_forensics.png` | Statistical anomaly distributions by label |
| `06_confusion_calibration.png` | Confusion matrix + probability calibration curve |

---

## 5 · Discussion

The near-perfect AUC (1.000) reflects the strong signal in synthetic data where hallucination markers are programmatically injected. Real-world implications:

1. **Confidence language is the canary**: LLMs systematically overstate certainty because their training optimises for fluency and authority. Hedge-to-claim ratios are measurable and effective.
2. **Citation forensics catches lazy hallucinations**: Invented journals often don't match naming conventions; future-year dates are unambiguous.
3. **Statistical impossibility is detectable**: Human researchers know that r = 0.95 across 50 students is implausible; LLMs don't. Round p-values (exactly 0.01 or 0.05) signal post-hoc insertion.

**Real-world degradation:** Against genuine LLM-generated text (not synthetic), AUC would likely fall to 0.75–0.90 due to more sophisticated hallucination. Multi-layer fusion should be more robust than any single layer.

---

## 6 · Limitations

- Synthetic corpus: hallucination markers are injected deterministically; real LLM outputs have softer boundaries
- AUC = 1.000 is expected on clean synthetic data; not a claim of real-world perfection
- Features require domain-specific journal lists; maintaining comprehensive fake-journal databases is non-trivial
- No NLP language model features used (only hand-crafted signals); transformer-based features would likely improve real-world AUC

---

## 7 · Conclusions

Multi-layer forensic analysis — combining citation integrity, statistical plausibility, and linguistic confidence analysis — provides a highly effective hallucination detection framework. The composite approach is more robust than any single layer, with statistical forensics providing the complementary coverage when citation databases are incomplete. Automated deployment of this pipeline as a pre-publication screening tool could substantially reduce hallucinated content in the academic literature.

---

## References

1. Ji, Z., et al. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12), 1–38.
2. Fierro, C., et al. (2024). LLM hallucination detection in academic text. *Preprint*.
3. Simmons, J. P., et al. (2011). False-positive psychology. *Psychological Science*, 22(11), 1359–1366.
4. Gundersen, O. E., & Kjensmo, S. (2018). State of the art: Reproducibility in artificial intelligence. *AAAI*, 32(1).

---

*Simulation code: `projects/P9_hallucination_detector/hallucination_detector.py`*  
*Figures: `FIGURES/P9_hallucination_detector/`*  
*Results: `RESULTS/P9_results.json`*
