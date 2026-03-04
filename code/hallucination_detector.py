"""
P9: AI Hallucination Detector for Academic Writing
BT33 Implementation: Multi-layer heuristic + statistical hallucination detection
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import re, json, os, warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
COLORS = ["#2196F3","#E91E63","#4CAF50","#FF9800","#9C27B0","#00BCD4"]
OUT = os.path.join(os.path.dirname(__file__), "figures")
RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES, exist_ok=True)

print("=" * 60)
print("  P9: AI HALLUCINATION DETECTOR")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC CORPUS GENERATION
# ─────────────────────────────────────────────────────────────────────────────
N_PAPERS = 600
HALLUCINATION_RATE = 0.45   # 45% of papers contain AI hallucinations

print(f"\n[1] Generating synthetic corpus: {N_PAPERS} papers ({HALLUCINATION_RATE*100:.0f}% hallucinated)...")

# Real journal names and fake ones
REAL_JOURNALS = ["Nature", "Science", "JAMA", "Lancet", "PNAS", "Cell",
                 "Nature Neuroscience", "Psychological Science", "PLOS ONE",
                 "Journal of Educational Psychology", "Cognition", "NeuroImage",
                 "British Medical Journal", "New England Journal of Medicine"]
FAKE_JOURNALS = ["Journal of Advanced Cognitive Neuroscience",
                 "International Review of Learning Sciences",
                 "European Mind & Brain Research",
                 "Global Journal of Educational Intelligence",
                 "Frontiers in Neural Computing Research",
                 "Advanced Studies in Psychological Technology"]

# Reference templates
LAST_NAMES = ["Johnson","Williams","Smith","Brown","Davis","Miller","Wilson",
              "Moore","Taylor","Anderson","Thomas","Jackson","White","Harris",
              "Martin","Garcia","Martinez","Lee","Thompson","Clark"]
YEARS_VALID = list(range(2015, 2026))
YEARS_FUTURE = list(range(2026, 2035))

def gen_author(hallucinated=False):
    n = np.random.choice(LAST_NAMES)
    if hallucinated and np.random.rand() < 0.12:
        # Ghost author — plausible sounding but doesn't publish on topic
        return f"X. {n}"
    elif not hallucinated and np.random.rand() < 0.02:
        # Occasionally a real paper has an unusual author format
        return f"X. {n}"
    return f"{n[0]}. {n}"

def gen_citation(hallucinated=False):
    n_authors = np.random.randint(1, 5)
    authors   = ", ".join([gen_author(hallucinated) for _ in range(n_authors)])
    r = np.random.rand()
    if hallucinated and r < 0.15:
        year = np.random.choice(YEARS_FUTURE)   # future year (reduced from 40%)
    elif not hallucinated and r < 0.02:
        year = np.random.choice(YEARS_FUTURE)   # rare preprint date error in real papers
    elif hallucinated and r < 0.30:
        year = np.random.randint(1950, 2000)     # implausibly old for topic
    else:
        year = np.random.choice(YEARS_VALID)
    # Realistic noise: hallucinated papers ~45% fake journals; real ~20%
    if hallucinated:
        journal = np.random.choice(FAKE_JOURNALS if np.random.rand() < 0.45 else REAL_JOURNALS)
    else:
        journal = np.random.choice(FAKE_JOURNALS if np.random.rand() < 0.20 else REAL_JOURNALS)
    vol     = np.random.randint(1, 50)
    page    = np.random.randint(1, 999)
    return {"authors": authors, "year": int(year), "journal": journal,
            "volume": vol, "page": page, "is_hallucinated": hallucinated}

def gen_statistic(hallucinated=False):
    """Generate an in-text statistic (percentage, correlation, p-value, etc.)"""
    # Realistic: not all hallucinated stats are suspicious; real papers can have
    # round numbers, coincidentally high correlations, or small p-values
    if hallucinated:
        # 35% of hallucinated stats are suspicious (reduced from 55%)
        actually_suspicious = np.random.rand() < 0.35
    else:
        # 15% of real stats happen to look suspicious (raised from 10%)
        actually_suspicious = np.random.rand() < 0.15

    if actually_suspicious:
        # Implausible or precisely round numbers
        stat_type = np.random.choice(["pct", "corr", "pval", "effect"])
        if stat_type == "pct":
            val = np.random.choice([round(v, 0) for v in [12, 23, 34, 45, 56, 67, 78, 89, 43, 57]])
            return {"type": "percentage", "value": float(val), "is_hallucinated": True,
                    "suspicion": "round_number"}
        elif stat_type == "corr":
            val = round(np.random.uniform(0.85, 0.99), 2)
            return {"type": "correlation", "value": val, "is_hallucinated": True,
                    "suspicion": "implausibly_high"}
        elif stat_type == "pval":
            val = round(np.random.choice([0.001, 0.0001, 0.00001, 0.000001]), 6)
            return {"type": "p_value", "value": float(val), "is_hallucinated": True,
                    "suspicion": "suspiciously_small"}
        else:
            val = round(np.random.uniform(1.5, 3.5), 1)
            return {"type": "effect_size", "value": val, "is_hallucinated": True,
                    "suspicion": "large_effect"}
    else:
        stat_type = np.random.choice(["pct", "corr", "pval", "effect"])
        if stat_type == "pct":
            val = round(np.random.uniform(10, 85) + np.random.randn() * 5, 1)
            return {"type": "percentage", "value": val, "is_hallucinated": False, "suspicion": "none"}
        elif stat_type == "corr":
            val = round(np.random.uniform(0.2, 0.75), 2)
            return {"type": "correlation", "value": val, "is_hallucinated": False, "suspicion": "none"}
        elif stat_type == "pval":
            val = round(np.random.uniform(0.01, 0.05), 3)
            return {"type": "p_value", "value": val, "is_hallucinated": False, "suspicion": "none"}
        else:
            val = round(np.random.uniform(0.2, 1.2), 2)
            return {"type": "effect_size", "value": val, "is_hallucinated": False, "suspicion": "none"}

CONFIDENCE_PHRASES_HIGH = [
    "It is well established that", "As demonstrated conclusively by",
    "The landmark study by", "All researchers agree that",
    "Definitively proven by", "The seminal work of"
]
CONFIDENCE_PHRASES_MED = [
    "Studies suggest that", "Evidence indicates", "Research shows",
    "Findings suggest", "Results demonstrate", "Analysis reveals"
]

papers = []
for pid in range(N_PAPERS):
    is_hallucinated = np.random.rand() < HALLUCINATION_RATE
    n_citations     = np.random.randint(8, 25)
    n_stats         = np.random.randint(3, 10)

    # For hallucinated papers: mix of real and fake citations
    citations = []
    if is_hallucinated:
        n_fake = max(1, int(n_citations * np.random.uniform(0.3, 0.8)))
        citations = [gen_citation(hallucinated=i < n_fake) for i in range(n_citations)]
        np.random.shuffle(citations)
    else:
        citations = [gen_citation(hallucinated=False) for _ in range(n_citations)]

    statistics = [gen_statistic(hallucinated=is_hallucinated)
                  for _ in range(n_stats)]

    # High-confidence language: overlapping Normal distributions (realistic)
    # Hallucinated papers trend higher but with same variance → substantial overlap
    if is_hallucinated:
        confidence_phrase_count = np.random.randint(0, 6)
        high_conf_frac = float(np.clip(np.random.normal(0.55, 0.22), 0.0, 1.0))
    else:
        confidence_phrase_count = np.random.randint(0, 5)
        high_conf_frac = float(np.clip(np.random.normal(0.42, 0.22), 0.0, 1.0))

    papers.append({
        "id": pid,
        "is_hallucinated": is_hallucinated,
        "citations": citations,
        "statistics": statistics,
        "confidence_phrase_count": confidence_phrase_count,
        "high_conf_frac": high_conf_frac,
        "n_citations": n_citations,
        "n_stats": n_stats
    })

n_hallucinated = sum(p["is_hallucinated"] for p in papers)
print(f"    Papers: {N_PAPERS} total, {n_hallucinated} hallucinated ({n_hallucinated/N_PAPERS*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DETECTION MODULE 1: CITATION FORENSICS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2] Citation forensics analysis...")

def analyse_citations(paper):
    cites = paper["citations"]
    n_fake_journals = sum(1 for c in cites if c["journal"] in FAKE_JOURNALS)
    n_future_years  = sum(1 for c in cites if c["year"] > 2025)
    n_plausible     = sum(1 for c in cites if c["journal"] in REAL_JOURNALS and c["year"] <= 2025)
    n_ghost_marker  = sum(1 for c in cites if c["authors"].startswith("X."))
    frac_fake_j     = n_fake_journals / max(1, len(cites))
    frac_future     = n_future_years  / max(1, len(cites))
    frac_real       = n_plausible     / max(1, len(cites))
    citation_score  = (frac_fake_j * 0.5 + frac_future * 0.3 +
                       (1 - frac_real) * 0.1 + n_ghost_marker / max(1, len(cites)) * 0.1)
    return {
        "n_fake_journals": n_fake_journals, "n_future_years": n_future_years,
        "n_plausible": n_plausible, "n_ghost_markers": n_ghost_marker,
        "frac_fake_journals": frac_fake_j, "frac_future_years": frac_future,
        "citation_suspicion_score": max(0, min(1, citation_score))
    }

citation_features = [analyse_citations(p) for p in papers]

# ─────────────────────────────────────────────────────────────────────────────
# 3. DETECTION MODULE 2: STATISTICAL FORENSICS
# ─────────────────────────────────────────────────────────────────────────────
print(f"[3] Statistical forensics analysis...")

def analyse_statistics(paper):
    stats_list = paper["statistics"]
    n_round    = sum(1 for s in stats_list if s.get("suspicion") == "round_number")
    n_high_corr = sum(1 for s in stats_list if s.get("suspicion") == "implausibly_high")
    n_tiny_p    = sum(1 for s in stats_list if s.get("suspicion") == "suspiciously_small")
    n_large_eff = sum(1 for s in stats_list if s.get("suspicion") == "large_effect")
    n_total     = max(1, len(stats_list))

    # Value clustering: hallucinated stats are often round numbers
    values = [s["value"] for s in stats_list if s["type"] == "percentage"]
    round_frac = sum(1 for v in values if v == round(v, 0)) / max(1, len(values))

    stat_score  = ((n_round + n_high_corr + n_tiny_p + n_large_eff) / n_total * 0.6 +
                    round_frac * 0.4)
    return {
        "n_suspicious_stats": n_round + n_high_corr + n_tiny_p + n_large_eff,
        "round_number_frac": round_frac,
        "stat_suspicion_score": max(0, min(1, stat_score))
    }

stat_features = [analyse_statistics(p) for p in papers]

# ─────────────────────────────────────────────────────────────────────────────
# 4. DETECTION MODULE 3: CONFIDENCE-ACCURACY MISMATCH
# ─────────────────────────────────────────────────────────────────────────────
print(f"[4] Confidence-accuracy mismatch detection...")

confidence_scores = np.array([
    p["high_conf_frac"] * 0.6 + p["confidence_phrase_count"] / 10 * 0.4
    for p in papers
])
confidence_scores = np.clip(confidence_scores, 0, 1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. COMPOSITE FEATURE MATRIX + ML DETECTION
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[5] Building feature matrix and training hallucination classifier...")

feature_rows = []
for i, p in enumerate(papers):
    cf = citation_features[i]
    sf = stat_features[i]
    row = {
        "n_citations":            p["n_citations"],
        "n_stats":                p["n_stats"],
        "frac_fake_journals":     cf["frac_fake_journals"],
        "frac_future_years":      cf["frac_future_years"],
        "n_ghost_markers":        cf["n_ghost_markers"],
        "citation_suspicion":     cf["citation_suspicion_score"],
        "n_suspicious_stats":     sf["n_suspicious_stats"],
        "round_number_frac":      sf["round_number_frac"],
        "stat_suspicion":         sf["stat_suspicion_score"],
        "confidence_score":       float(confidence_scores[i]),
        "high_conf_frac":         p["high_conf_frac"],
        "conf_phrase_count":      p["confidence_phrase_count"],
    }
    feature_rows.append(row)

df = pd.DataFrame(feature_rows)
df["label"] = [int(p["is_hallucinated"]) for p in papers]

X = df.drop("label", axis=1).values
y = df["label"].values

# Add realistic measurement noise (imperfect feature extraction, OCR errors, etc.)
rng_noise = np.random.RandomState(99)
X += rng_noise.randn(*X.shape) * 0.10 * (X.std(axis=0) + 1e-6)

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}

cv_results = {}
for name, model in models.items():
    X_in = X_sc if name == "Logistic Regression" else X
    aucs = cross_val_score(model, X_in, y, cv=skf, scoring='roc_auc')
    cv_results[name] = {"auc_mean": float(aucs.mean()), "auc_std": float(aucs.std())}
    print(f"    {name}: AUC = {aucs.mean():.3f} ± {aucs.std():.3f}")

# Final model: Random Forest on 80/20 split
split = int(0.8 * len(y))
idx   = np.random.permutation(len(y))
tr, te = idx[:split], idx[split:]

rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
rf_final.fit(X[tr], y[tr])
y_prob = rf_final.predict_proba(X[te])[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
final_auc = roc_auc_score(y[te], y_prob)
final_ap  = average_precision_score(y[te], y_prob)
report    = classification_report(y[te], y_pred, output_dict=True)
final_f1  = report["weighted avg"]["f1-score"]
final_acc = report["accuracy"]

print(f"\n    Final RF: AUC={final_auc:.3f}, AP={final_ap:.3f}, F1={final_f1:.3f}, Acc={final_acc:.3f}")

# Feature importances
feat_names = list(df.drop("label", axis=1).columns)
importances = rf_final.feature_importances_

# ─────────────────────────────────────────────────────────────────────────────
# 6. HALLUCINATION SCORE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
# Composite hallucination score
hallucination_scores = (
    np.array([cf["citation_suspicion_score"] for cf in citation_features]) * 0.45 +
    np.array([sf["stat_suspicion_score"]     for sf in stat_features])     * 0.35 +
    confidence_scores                                                        * 0.20
)

# Per-detection-layer results
layer_aucs = {
    "Citation forensics": roc_auc_score(y, [cf["citation_suspicion_score"] for cf in citation_features]),
    "Statistical forensics": roc_auc_score(y, [sf["stat_suspicion_score"] for sf in stat_features]),
    "Confidence mismatch": roc_auc_score(y, confidence_scores),
    "Composite score": roc_auc_score(y, hallucination_scores),
    "Random Forest (ML)": roc_auc_score(y, rf_final.predict_proba(X)[:, 1])
}

print(f"\n    Layer-by-layer AUC:")
for layer, auc in layer_aucs.items():
    print(f"      {layer:30}: {auc:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[7] Generating figures...")

# FIG 1: Detection pipeline overview
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("AI Hallucination Detector: Multi-Layer Detection Pipeline", fontsize=14, fontweight='bold')
axes = axes.ravel()

# Hallucination score distribution by group
scores_hall = hallucination_scores[y == 1]
scores_real = hallucination_scores[y == 0]
axes[0].hist(scores_real, bins=30, alpha=0.7, color=COLORS[2], label=f"Genuine (n={sum(y==0)})")
axes[0].hist(scores_hall, bins=30, alpha=0.7, color=COLORS[1], label=f"Hallucinated (n={sum(y==1)})")
axes[0].set_title("Composite Hallucination Score Distribution")
axes[0].set_xlabel("Hallucination Score")
axes[0].set_ylabel("Count")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Citation suspicion by group
c_real = [cf["citation_suspicion_score"] for cf, p in zip(citation_features, papers) if not p["is_hallucinated"]]
c_hall = [cf["citation_suspicion_score"] for cf, p in zip(citation_features, papers) if p["is_hallucinated"]]
axes[1].boxplot([c_real, c_hall], labels=["Genuine", "Hallucinated"],
                patch_artist=True,
                boxprops=dict(facecolor=COLORS[0], alpha=0.6))
axes[1].set_title("Citation Forensics Score by Group")
axes[1].set_ylabel("Citation Suspicion Score")
axes[1].grid(alpha=0.3)

# Statistical suspicion by group
s_real = [sf["stat_suspicion_score"] for sf, p in zip(stat_features, papers) if not p["is_hallucinated"]]
s_hall = [sf["stat_suspicion_score"] for sf, p in zip(stat_features, papers) if p["is_hallucinated"]]
axes[2].boxplot([s_real, s_hall], labels=["Genuine", "Hallucinated"],
                patch_artist=True,
                boxprops=dict(facecolor=COLORS[4], alpha=0.6))
axes[2].set_title("Statistical Forensics Score by Group")
axes[2].set_ylabel("Statistical Suspicion Score")
axes[2].grid(alpha=0.3)

# Confidence score
conf_real = confidence_scores[y == 0]
conf_hall = confidence_scores[y == 1]
axes[3].hist(conf_real, bins=25, alpha=0.7, color=COLORS[2], label="Genuine")
axes[3].hist(conf_hall, bins=25, alpha=0.7, color=COLORS[1], label="Hallucinated")
axes[3].set_title("Confidence-Accuracy Mismatch Score")
axes[3].set_xlabel("High-confidence language score")
axes[3].set_ylabel("Count"); axes[3].legend(); axes[3].grid(alpha=0.3)

# Layer-by-layer AUC comparison
layer_names = list(layer_aucs.keys())
layer_vals  = list(layer_aucs.values())
bar_cl = [COLORS[i % len(COLORS)] for i in range(len(layer_names))]
bars = axes[4].barh(layer_names, layer_vals, color=bar_cl, alpha=0.85, edgecolor='white')
for bar, val in zip(bars, layer_vals):
    axes[4].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va='center', fontsize=9, fontweight='bold')
axes[4].set_xlim(0.5, 1.05)
axes[4].axvline(0.5, color='gray', linestyle='--', lw=1, label="Random (0.5)")
axes[4].set_title("Detection AUC by Layer")
axes[4].set_xlabel("ROC-AUC")
axes[4].legend(fontsize=8); axes[4].grid(axis='x', alpha=0.3)

# Cross-validation AUC
cv_names  = list(cv_results.keys())
cv_means  = [cv_results[n]["auc_mean"] for n in cv_names]
cv_stds   = [cv_results[n]["auc_std"]  for n in cv_names]
x_pos = range(len(cv_names))
bars2 = axes[5].bar(cv_names, cv_means, color=[COLORS[i] for i in range(len(cv_names))],
                    alpha=0.8, edgecolor='white')
axes[5].errorbar(x_pos, cv_means, yerr=cv_stds, fmt='none', color='black', capsize=5, lw=2)
for bar, m in zip(bars2, cv_means):
    axes[5].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{m:.3f}", ha='center', fontsize=9, fontweight='bold')
axes[5].set_title("5-Fold CV ROC-AUC by Model")
axes[5].set_ylabel("AUC"); axes[5].set_ylim(0.5, 1.05)
axes[5].set_xticklabels(cv_names, fontsize=9, rotation=15)
axes[5].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/01_detection_pipeline.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Fig 1 saved")

# FIG 2: ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Hallucination Detector: ROC and Precision-Recall Curves", fontsize=13, fontweight='bold')

# ROC
fpr, tpr, _ = roc_curve(y[te], y_prob)
axes[0].plot(fpr, tpr, color=COLORS[0], lw=2.5, label=f"Random Forest (AUC={final_auc:.3f})")
fpr_c, tpr_c, _ = roc_curve(y, hallucination_scores)
auc_c = roc_auc_score(y, hallucination_scores)
axes[0].plot(fpr_c, tpr_c, color=COLORS[1], lw=2, linestyle='--', label=f"Composite score (AUC={auc_c:.3f})")
axes[0].plot([0,1],[0,1],'gray',linestyle=':',lw=1.5)
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

# PR
prec, rec, _ = precision_recall_curve(y[te], y_prob)
axes[1].plot(rec, prec, color=COLORS[0], lw=2.5, label=f"RF (AP={final_ap:.3f})")
prec_c, rec_c, _ = precision_recall_curve(y, hallucination_scores)
ap_c = average_precision_score(y, hallucination_scores)
axes[1].plot(rec_c, prec_c, color=COLORS[1], lw=2, linestyle='--', label=f"Composite (AP={ap_c:.3f})")
axes[1].axhline(y.mean(), color='gray', linestyle=':', lw=1.5, label=f"Baseline ({y.mean():.2f})")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve"); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/02_roc_pr_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Fig 2 saved")

# FIG 3: Feature importances
fig, ax = plt.subplots(figsize=(10, 7))
sorted_idx = np.argsort(importances)
ax.barh(np.array(feat_names)[sorted_idx], importances[sorted_idx],
        color=[COLORS[i % len(COLORS)] for i in range(len(feat_names))], alpha=0.85, edgecolor='white')
ax.set_xlabel("Feature Importance (Random Forest)")
ax.set_title("Feature Importances: Hallucination Detection\nWhich signals betray AI-generated content?", fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/03_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Fig 3 saved")

# FIG 4: Citation forensics deep dive
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Citation Forensics: Anatomy of Hallucinated References", fontsize=13, fontweight='bold')

# Fake journal fraction
fkj_r = [cf["frac_fake_journals"] for cf, p in zip(citation_features, papers) if not p["is_hallucinated"]]
fkj_h = [cf["frac_fake_journals"] for cf, p in zip(citation_features, papers) if p["is_hallucinated"]]
axes[0].boxplot([fkj_r, fkj_h], labels=["Genuine","Hallucinated"], patch_artist=True,
                boxprops=dict(facecolor=COLORS[0], alpha=0.6))
axes[0].set_title("Fraction from Unknown Journals")
axes[0].set_ylabel("Fraction fake journals")
axes[0].grid(alpha=0.3)

# Future year citations
fty_r = [cf["frac_future_years"] for cf, p in zip(citation_features, papers) if not p["is_hallucinated"]]
fty_h = [cf["frac_future_years"] for cf, p in zip(citation_features, papers) if p["is_hallucinated"]]
axes[1].boxplot([fty_r, fty_h], labels=["Genuine","Hallucinated"], patch_artist=True,
                boxprops=dict(facecolor=COLORS[4], alpha=0.6))
axes[1].set_title("Future-Year Citations (>2025)")
axes[1].set_ylabel("Fraction with future years")
axes[1].grid(alpha=0.3)

# Ghost author markers
gh_r = [cf["n_ghost_markers"] for cf, p in zip(citation_features, papers) if not p["is_hallucinated"]]
gh_h = [cf["n_ghost_markers"] for cf, p in zip(citation_features, papers) if p["is_hallucinated"]]
axes[2].boxplot([gh_r, gh_h], labels=["Genuine","Hallucinated"], patch_artist=True,
                boxprops=dict(facecolor=COLORS[1], alpha=0.6))
axes[2].set_title("Ghost Author Markers")
axes[2].set_ylabel("Count of ghost markers")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/04_citation_forensics.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Fig 4 saved")

# FIG 5: Statistical forensics
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Statistical Forensics: Anomalous Numbers in AI-Generated Text", fontsize=13, fontweight='bold')

# Round number fraction
rn_r = [sf["round_number_frac"] for sf, p in zip(stat_features, papers) if not p["is_hallucinated"]]
rn_h = [sf["round_number_frac"] for sf, p in zip(stat_features, papers) if p["is_hallucinated"]]
axes[0].boxplot([rn_r, rn_h], labels=["Genuine","Hallucinated"], patch_artist=True,
                boxprops=dict(facecolor=COLORS[2], alpha=0.6))
axes[0].set_title("Round Number Fraction\n(Benford's law violation)")
axes[0].set_ylabel("Fraction round numbers")
axes[0].grid(alpha=0.3)

# Suspicious stat count
sn_r = [sf["n_suspicious_stats"] for sf, p in zip(stat_features, papers) if not p["is_hallucinated"]]
sn_h = [sf["n_suspicious_stats"] for sf, p in zip(stat_features, papers) if p["is_hallucinated"]]
axes[1].hist(sn_r, bins=12, alpha=0.7, color=COLORS[2], label="Genuine")
axes[1].hist(sn_h, bins=12, alpha=0.7, color=COLORS[1], label="Hallucinated")
axes[1].set_title("Suspicious Statistic Count Per Paper")
axes[1].set_xlabel("N suspicious values"); axes[1].set_ylabel("Count")
axes[1].legend(); axes[1].grid(alpha=0.3)

# High-confidence language
axes[2].scatter(confidence_scores[y==0], [cf["frac_fake_journals"] for cf, lab in zip(citation_features, y) if lab==0],
                s=10, alpha=0.5, color=COLORS[2], label="Genuine")
axes[2].scatter(confidence_scores[y==1], [cf["frac_fake_journals"] for cf, lab in zip(citation_features, y) if lab==1],
                s=10, alpha=0.5, color=COLORS[1], label="Hallucinated")
axes[2].set_xlabel("Confidence-language score")
axes[2].set_ylabel("Fraction fake journals")
axes[2].set_title("Confidence × Citation Space\n(separation between groups)")
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/05_statistical_forensics.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Fig 5 saved")

# FIG 6: Confusion matrix + score calibration
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Final Model Performance: Confusion Matrix & Score Calibration", fontsize=13, fontweight='bold')

cm = confusion_matrix(y[te], y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Genuine","Hallucinated"])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f"Confusion Matrix (test set)\nACC={final_acc:.3f}, F1={final_f1:.3f}")

# Score calibration: predicted prob vs actual positive rate
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_acc   = []
bin_conf  = []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = (y_prob >= lo) & (y_prob < hi)
    if mask.sum() > 0:
        bin_acc.append(y[te][mask].mean())
        bin_conf.append(y_prob[mask].mean())
bin_acc  = np.array(bin_acc)
bin_conf = np.array(bin_conf)
axes[1].plot([0,1],[0,1],'k--',lw=1.5,label="Perfect calibration")
axes[1].plot(bin_conf, bin_acc, 'o-', color=COLORS[0], lw=2, markersize=8, label="RF classifier")
axes[1].fill_between([0,1],[0,0],[1,1],alpha=0.05,color='gray')
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Fraction positives (hallucinated)")
axes[1].set_title("Calibration Curve")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/06_confusion_calibration.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Fig 6 saved")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
results_out = {
    "corpus": {"n_papers": N_PAPERS, "hallucination_rate": HALLUCINATION_RATE,
               "n_hallucinated": int(n_hallucinated)},
    "cv_results": cv_results,
    "final_model": {"auc": float(final_auc), "average_precision": float(final_ap),
                    "f1_weighted": float(final_f1), "accuracy": float(final_acc)},
    "layer_aucs": layer_aucs,
    "feature_importances": dict(zip(feat_names, importances.tolist())),
    "top_features": [feat_names[i] for i in np.argsort(importances)[::-1][:5]]
}
with open(f"{RES}/results.json", "w") as f:
    json.dump(results_out, f, indent=2)

print(f"\n{'='*60}")
print(f"  P9 COMPLETE")
print(f"  Corpus: {N_PAPERS} papers, {HALLUCINATION_RATE*100:.0f}% hallucinated")
print(f"  Final RF AUC:  {final_auc:.3f}")
print(f"  Final RF AP:   {final_ap:.3f}")
print(f"  Final RF F1:   {final_f1:.3f}")
print(f"  Composite AUC: {auc_c:.3f}")
print(f"  Top feature:   {feat_names[np.argmax(importances)]}")
print(f"{'='*60}")
