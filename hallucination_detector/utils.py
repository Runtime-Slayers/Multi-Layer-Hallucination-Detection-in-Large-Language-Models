"""
Utility functions for results serialization and plotting.
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score
)

from .citation import analyse_citations
from .statistics import analyse_statistics
from .confidence import calculate_confidence_score

COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


def save_results(results, file_path):
    """Serialize the results dictionary to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def generate_figures(papers, eval_res, output_dir, mode="synthetic"):
    """
    Generate validation and diagnostic plots.
    Adapts automatically to CSV mode (3 features) or paper-based modes (12 features).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    y = eval_res["y"]
    te = eval_res["test_indices"]
    y_prob = eval_res["y_prob"]
    y_pred = eval_res["y_pred"]
    cv_results = eval_res["cv_results"]
    feat_names = eval_res["feature_names"]
    importances = eval_res["importances"]
    
    # 1. FIG 2: ROC and PR curves (Common for all modes)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Hallucination Detector ({mode.upper()}): ROC and Precision-Recall Curves", fontsize=13, fontweight='bold')
    
    fpr, tpr, _ = roc_curve(y[te], y_prob)
    axes[0].plot(fpr, tpr, color=COLORS[0], lw=2.5, label=f"Random Forest (AUC={eval_res['final_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], 'gray', linestyle=':', lw=1.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    prec, rec, _ = precision_recall_curve(y[te], y_prob)
    axes[1].plot(rec, prec, color=COLORS[0], lw=2.5, label=f"RF (AP={eval_res['final_ap']:.3f})")
    axes[1].axhline(y.mean(), color='gray', linestyle=':', lw=1.5, label=f"Baseline ({y.mean():.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_roc_pr_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. FIG 3: Feature importances (Common for all modes)
    fig, ax = plt.subplots(figsize=(10, 7))
    sorted_idx = np.argsort(importances)
    ax.barh(np.array(feat_names)[sorted_idx], importances[sorted_idx],
            color=[COLORS[i % len(COLORS)] for i in range(len(feat_names))], alpha=0.85, edgecolor='white')
    ax.set_xlabel("Feature Importance (Random Forest)")
    ax.set_title(f"Feature Importances ({mode.upper()})\nWhich signals betray AI-generated content?", fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_feature_importance.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. FIG 6: Confusion Matrix & Calibration (Common for all modes)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Final Model Performance ({mode.upper()}): Confusion Matrix & Score Calibration", fontsize=13, fontweight='bold')
    
    cm = confusion_matrix(y[te], y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Hallucinated"])
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title(f"Confusion Matrix (test set)\nACC={eval_res['final_acc']:.3f}, F1={eval_res['final_f1']:.3f}")
    
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc = []
    bin_conf = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            bin_acc.append(y[te][mask].mean())
            bin_conf.append(y_prob[mask].mean())
            
    bin_acc = np.array(bin_acc)
    bin_conf = np.array(bin_conf)
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, label="Perfect calibration")
    if len(bin_conf) > 0:
        axes[1].plot(bin_conf, bin_acc, 'o-', color=COLORS[0], lw=2, markersize=8, label="RF classifier")
    axes[1].fill_between([0, 1], [0, 0], [1, 1], alpha=0.05, color='gray')
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Fraction positives (hallucinated)")
    axes[1].set_title("Calibration Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_confusion_calibration.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Paper-based specific plots (require papers list)
    if mode in ("synthetic", "online") and papers is not None:
        citation_features = [analyse_citations(p) for p in papers]
        stat_features = [analyse_statistics(p) for p in papers]
        confidence_scores = np.array([calculate_confidence_score(p) for p in papers])
        
        # Composite score
        hallucination_scores = (
            np.array([cf["citation_suspicion_score"] for cf in citation_features]) * 0.45 +
            np.array([sf["stat_suspicion_score"] for sf in stat_features]) * 0.35 +
            confidence_scores * 0.20
        )
        
        # Layer AUCs
        layer_aucs = {
            "Citation forensics": float(roc_auc_score(y, [cf["citation_suspicion_score"] for cf in citation_features])),
            "Statistical forensics": float(roc_auc_score(y, [sf["stat_suspicion_score"] for sf in stat_features])),
            "Confidence mismatch": float(roc_auc_score(y, confidence_scores)),
            "Composite score": float(roc_auc_score(y, hallucination_scores)),
            "Random Forest (ML)": float(roc_auc_score(y, eval_res["rf_final"].predict_proba(eval_res["X_noise"])[:, 1]))
        }
        
        # FIG 1: Pipeline Overview
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle("AI Hallucination Detector: Multi-Layer Detection Pipeline", fontsize=14, fontweight='bold')
        axes = axes.ravel()
        
        scores_hall = hallucination_scores[y == 1]
        scores_real = hallucination_scores[y == 0]
        axes[0].hist(scores_real, bins=30, alpha=0.7, color=COLORS[2], label=f"Genuine (n={sum(y==0)})")
        axes[0].hist(scores_hall, bins=30, alpha=0.7, color=COLORS[1], label=f"Hallucinated (n={sum(y==1)})")
        axes[0].set_title("Composite Hallucination Score Distribution")
        axes[0].set_xlabel("Hallucination Score")
        axes[0].set_ylabel("Count")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        c_real = [cf["citation_suspicion_score"] for cf, p in zip(citation_features, papers) if not p["is_hallucinated"]]
        c_hall = [cf["citation_suspicion_score"] for cf, p in zip(citation_features, papers) if p["is_hallucinated"]]
        axes[1].boxplot([c_real, c_hall], labels=["Genuine", "Hallucinated"], patch_artist=True,
                        boxprops=dict(facecolor=COLORS[0], alpha=0.6))
        axes[1].set_title("Citation Forensics Score by Group")
        axes[1].set_ylabel("Citation Suspicion Score")
        axes[1].grid(alpha=0.3)
        
        s_real = [sf["stat_suspicion_score"] for sf, p in zip(stat_features, papers) if not p["is_hallucinated"]]
        s_hall = [sf["stat_suspicion_score"] for sf, p in zip(stat_features, papers) if p["is_hallucinated"]]
        axes[2].boxplot([s_real, s_hall], labels=["Genuine", "Hallucinated"], patch_artist=True,
                        boxprops=dict(facecolor=COLORS[4], alpha=0.6))
        axes[2].set_title("Statistical Forensics Score by Group")
        axes[2].set_ylabel("Statistical Suspicion Score")
        axes[2].grid(alpha=0.3)
        
        conf_real = confidence_scores[y == 0]
        conf_hall = confidence_scores[y == 1]
        axes[3].hist(conf_real, bins=25, alpha=0.7, color=COLORS[2], label="Genuine")
        axes[3].hist(conf_hall, bins=25, alpha=0.7, color=COLORS[1], label="Hallucinated")
        axes[3].set_title("Confidence-Accuracy Mismatch Score")
        axes[3].set_xlabel("High-confidence language score")
        axes[3].set_ylabel("Count")
        axes[3].legend()
        axes[3].grid(alpha=0.3)
        
        layer_names = list(layer_aucs.keys())
        layer_vals = list(layer_aucs.values())
        bar_cl = [COLORS[i % len(COLORS)] for i in range(len(layer_names))]
        bars = axes[4].barh(layer_names, layer_vals, color=bar_cl, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, layer_vals):
            axes[4].text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                         va='center', fontsize=9, fontweight='bold')
        axes[4].set_xlim(0.5, 1.05)
        axes[4].axvline(0.5, color='gray', linestyle='--', lw=1, label="Random (0.5)")
        axes[4].set_title("Detection AUC by Layer")
        axes[4].set_xlabel("ROC-AUC")
        axes[4].legend(fontsize=8)
        axes[4].grid(axis='x', alpha=0.3)
        
        cv_names = list(cv_results.keys())
        cv_means = [cv_results[n]["auc_mean"] for n in cv_names]
        cv_stds = [cv_results[n]["auc_std"] for n in cv_names]
        x_pos = range(len(cv_names))
        bars2 = axes[5].bar(cv_names, cv_means, color=[COLORS[i] for i in range(len(cv_names))],
                            alpha=0.8, edgecolor='white')
        axes[5].errorbar(x_pos, cv_means, yerr=cv_stds, fmt='none', color='black', capsize=5, lw=2)
        for bar, m in zip(bars2, cv_means):
            axes[5].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{m:.3f}",
                         ha='center', fontsize=9, fontweight='bold')
        axes[5].set_title("5-Fold CV ROC-AUC by Model")
        axes[5].set_ylabel("AUC")
        axes[5].set_ylim(0.5, 1.05)
        axes[5].set_xticks(list(x_pos))
        axes[5].set_xticklabels(cv_names, fontsize=9, rotation=15)
        axes[5].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "01_detection_pipeline.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # FIG 4: Citation forensics deep dive
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle("Citation Forensics: Anatomy of Hallucinated References", fontsize=13, fontweight='bold')
        
        fkj_r = [cf["frac_fake_journals"] for cf, p in zip(citation_features, papers) if not p["is_hallucinated"]]
        fkj_h = [cf["frac_fake_journals"] for cf, p in zip(citation_features, papers) if p["is_hallucinated"]]
        axes[0].boxplot([fkj_r, fkj_h], labels=["Genuine", "Hallucinated"], patch_artist=True,
                        boxprops=dict(facecolor=COLORS[0], alpha=0.6))
        axes[0].set_title("Fraction from Unknown Journals")
        axes[0].set_ylabel("Fraction fake journals")
        axes[0].grid(alpha=0.3)
        
        fty_r = [cf["frac_future_years"] for cf, p in zip(citation_features, papers) if not p["is_hallucinated"]]
        fty_h = [cf["frac_future_years"] for cf, p in zip(citation_features, papers) if p["is_hallucinated"]]
        axes[1].boxplot([fty_r, fty_h], labels=["Genuine", "Hallucinated"], patch_artist=True,
                        boxprops=dict(facecolor=COLORS[4], alpha=0.6))
        axes[1].set_title("Future-Year Citations (>2025)")
        axes[1].set_ylabel("Fraction with future years")
        axes[1].grid(alpha=0.3)
        
        gh_r = [cf["n_ghost_markers"] for cf, p in zip(citation_features, papers) if not p["is_hallucinated"]]
        gh_h = [cf["n_ghost_markers"] for cf, p in zip(citation_features, papers) if p["is_hallucinated"]]
        axes[2].boxplot([gh_r, gh_h], labels=["Genuine", "Hallucinated"], patch_artist=True,
                        boxprops=dict(facecolor=COLORS[1], alpha=0.6))
        axes[2].set_title("Ghost Author Markers")
        axes[2].set_ylabel("Count of ghost markers")
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "04_citation_forensics.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # FIG 5: Statistical forensics
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle("Statistical Forensics: Anomalous Numbers in AI-Generated Text", fontsize=13, fontweight='bold')
        
        rn_r = [sf["round_number_frac"] for sf, p in zip(stat_features, papers) if not p["is_hallucinated"]]
        rn_h = [sf["round_number_frac"] for sf, p in zip(stat_features, papers) if p["is_hallucinated"]]
        axes[0].boxplot([rn_r, rn_h], labels=["Genuine", "Hallucinated"], patch_artist=True,
                        boxprops=dict(facecolor=COLORS[2], alpha=0.6))
        axes[0].set_title("Round Number Fraction\n(Benford's law violation)")
        axes[0].set_ylabel("Fraction round numbers")
        axes[0].grid(alpha=0.3)
        
        sn_r = [sf["n_suspicious_stats"] for sf, p in zip(stat_features, papers) if not p["is_hallucinated"]]
        sn_h = [sf["n_suspicious_stats"] for sf, p in zip(stat_features, papers) if p["is_hallucinated"]]
        axes[1].hist(sn_r, bins=12, alpha=0.7, color=COLORS[2], label="Genuine")
        axes[1].hist(sn_h, bins=12, alpha=0.7, color=COLORS[1], label="Hallucinated")
        axes[1].set_title("Suspicious Statistic Count Per Paper")
        axes[1].set_xlabel("N suspicious values")
        axes[1].set_ylabel("Count")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        axes[2].scatter(confidence_scores[y == 0],
                        [cf["frac_fake_journals"] for cf, lab in zip(citation_features, y) if lab == 0],
                        s=10, alpha=0.5, color=COLORS[2], label="Genuine")
                        
        axes[2].scatter(confidence_scores[y == 1],
                        [cf["frac_fake_journals"] for cf, lab in zip(citation_features, y) if lab == 1],
                        s=10, alpha=0.5, color=COLORS[1], label="Hallucinated")
                        
        axes[2].set_xlabel("Confidence-language score")
        axes[2].set_ylabel("Fraction fake journals")
        axes[2].set_title("Confidence × Citation Space\n(separation between groups)")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "05_statistical_forensics.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        return layer_aucs
        
    return None
