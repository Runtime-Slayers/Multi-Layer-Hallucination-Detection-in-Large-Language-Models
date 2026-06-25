# Mathematical Foundations of the Multi-Layer Hallucination Detector

The academic hallucination detector utilizes three independent forensic layers to identify generated academic text containing fabricated references, impossible statistics, and overconfident language.

---

## 1. Citation Forensics Layer

The citation verification engine computes a citation suspicion score $S_{\text{cite}} \in [0, 1]$ for a paper based on the presence of structural citation anomalies:

$$S_{\text{cite}} = \max\left(0, \min\left(1, 0.5 \cdot F_{\text{fake}} + 0.3 \cdot F_{\text{future}} + 0.1 \cdot (1 - F_{\text{real}}) + 0.1 \cdot \frac{N_{\text{ghost}}}{N_{\text{total}}}\right)\right)$$

Where:
*   $N_{\text{total}}$ is the total number of cited works in the paper.
*   $F_{\text{fake}} = \frac{N_{\text{fake\_journals}}}{N_{\text{total}}}$ is the fraction of cited works published in known fake/predatory journals.
*   $F_{\text{future}} = \frac{N_{\text{future\_years}}}{N_{\text{total}}}$ is the fraction of cited works with publication years in the future (e.g., year $> 2025$).
*   $F_{\text{real}} = \frac{N_{\text{plausible\_journals}}}{N_{\text{total}}}$ is the fraction of verified real journal publications published in past years (year $\le 2025$).
*   $N_{\text{ghost}}$ is the count of citations referencing authors with unusual formats (e.g., starting with `"X."`), indicating ghost authorship anomalies.

---

## 2. Statistical Forensics Layer

The statistical forensics layer detects implausible numerical data using two features: statistical suspicion (extreme thresholds) and psychological round-number clustering (a violation of Benford's Law).

The statistical suspicion score $S_{\text{stat}} \in [0, 1]$ is defined as:

$$S_{\text{stat}} = 0.6 \cdot \frac{N_{\text{suspicious}}}{N_{\text{total\_stats}}} + 0.4 \cdot F_{\text{round\_percent}}$$

Where:
*   $N_{\text{total\_stats}}$ is the total count of statistical statements extracted from the paper.
*   $N_{\text{suspicious}}$ counts statistical values that violate realistic bounds:
    *   **Percentages**: precise integer round numbers $v \in \{12, 23, 34, 45, 56, 67, 78, 89, 43, 57\}$.
    *   **Correlations**: extremely high Pearson/Spearman coefficients ($r \in [0.85, 0.99]$) in fields characterized by noisy measurements.
    *   **Significance tests**: suspiciously low p-values ($p < 10^{-6}$) suggesting artificial precision.
    *   **Effect sizes**: implausibly large standard effect metrics (Cohen's $d > 1.5$).
*   $F_{\text{round\_percent}}$ represents the fraction of percentage values that are exactly round integers:
    $$F_{\text{round\_percent}} = \frac{\sum_{i=1}^{M_{\text{pct}}} \mathbb{I}(v_i = \lfloor v_i \rfloor)}{M_{\text{pct}}}$$
    where $M_{\text{pct}}$ is the count of percentage claims and $\mathbb{I}$ is the indicator function.

---

## 3. Confidence-Accuracy Mismatch Layer

LLM-generated academic text is characterized by overuse of high-certainty statements without appropriate epistemic hedging. The confidence score $S_{\text{conf}} \in [0, 1]$ is calculated as:

$$S_{\text{conf}} = \max\left(0, \min\left(1, 0.6 \cdot F_{\text{high\_conf}} + 0.4 \cdot \frac{N_{\text{conf\_phrases}}}{10}\right)\right)$$

Where:
*   $F_{\text{high\_conf}}$ is the fraction of sentences using high-confidence structures, modeled as:
    $$F_{\text{high\_conf}} \sim \begin{cases} 
      \mathcal{N}(0.55, 0.22) & \text{if paper is hallucinated} \\
      \mathcal{N}(0.42, 0.22) & \text{if paper is genuine}
    \end{cases}$$
*   $N_{\text{conf\_phrases}}$ is the count of strong confidence phrases (e.g., *"definitively proven by"*, *"all researchers agree that"*, *"as demonstrated conclusively by"*).

---

## 4. Machine Learning Integration

The features extracted from the three layers are concatenated into a 12-dimensional feature vector $\mathbf{x} \in \mathbb{R}^{12}$:

$$\mathbf{x} = \begin{bmatrix}
N_{\text{citations}} \\
N_{\text{stats}} \\
F_{\text{fake}} \\
F_{\text{future}} \\
N_{\text{ghost}} \\
S_{\text{cite}} \\
N_{\text{suspicious}} \\
F_{\text{round\_percent}} \\
S_{\text{stat}} \\
S_{\text{conf}} \\
F_{\text{high\_conf}} \\
N_{\text{conf\_phrases}}
\end{bmatrix}$$

These raw features are perturbed with measurement noise ($\sigma = 0.10 \cdot \sigma_j$) to simulate realistic text extraction limits:

$$\mathbf{x}_i' = \mathbf{x}_i + \boldsymbol{\epsilon}_i, \quad \boldsymbol{\epsilon}_{ij} \sim \mathcal{N}\left(0, \left(0.10 \cdot \text{std}(\mathbf{X}_{*, j})\right)^2\right)$$

The perturbed features are standardized and passed to a Random Forest classifier $f(\mathbf{x}') \to P(y=1|\mathbf{x}')$ which outputs the probability that the text contains LLM hallucinations.
