#!/usr/bin/env python3
"""
P9 Validation Experiment runner.
Generates synthetic corpus, loads local CSV datasets, or downloads HaluEval,
trains classifiers, and outputs diagnostics plots and JSON results.
"""

import argparse
import logging
import os
import sys
import numpy as np

# Reconfigure stream encodings to prevent Windows console encoding crashes
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("P9-Validation")

try:
    from hallucination_detector import (
        generate_corpus,
        train_and_evaluate,
        generate_figures,
        save_results,
        analyse_citations,
        analyse_statistics,
        calculate_confidence_score
    )
    from hallucination_detector.corpus import load_halueval_as_papers
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from hallucination_detector import (
        generate_corpus,
        train_and_evaluate,
        generate_figures,
        save_results,
        analyse_citations,
        analyse_statistics,
        calculate_confidence_score
    )
    from hallucination_detector.corpus import load_halueval_as_papers


def parse_args():
    parser = argparse.ArgumentParser(description="P9 Hallucination Detector Validation Experiment")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "csv", "online"],
                        help="Data source mode: synthetic, csv (local file), or online (HaluEval)")
    parser.add_argument("--dataset-csv", type=str, default="data/P9_llm_detection_dataset.csv",
                        help="Path to the local CSV dataset for 'csv' mode")
    parser.add_argument("--subset", type=str, default="qa", choices=["qa", "dialogue", "summarization"],
                        help="HaluEval dataset subset for 'online' mode")
    parser.add_argument("--n-papers", type=int, default=600, help="Number of papers to simulate or download")
    parser.add_argument("--hallucination-rate", type=float, default=0.45, help="Simulation hallucination rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="figures", help="Directory to save figures")
    parser.add_argument("--results-file", type=str, default="data/results.json", help="Path to save results JSON")
    parser.add_argument("--live-citation-check", action="store_true", help="Perform real CrossRef/Semantic Scholar API calls")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info(f"P9: Multi-Layer AI Hallucination Detector Validation (Mode: {args.mode.upper()})")
    logger.info("=" * 60)
    
    papers = None
    
    if args.mode == "csv":
        logger.info(f"Loading local CSV dataset from {args.dataset_csv} ...")
        if not os.path.exists(args.dataset_csv):
            logger.error(f"Dataset CSV file not found: {args.dataset_csv}")
            sys.exit(1)
        logger.info("Local dataset loaded successfully.")
    elif args.mode == "online":
        logger.info(f"Downloading/loading HaluEval online dataset (subset: {args.subset}, limit: {args.n_papers}) ...")
        cache_dir = os.path.join(os.path.dirname(args.results_file), "cache")
        papers = load_halueval_as_papers(cache_dir=cache_dir, subset=args.subset, limit=args.n_papers)
        n_hall = sum(p["is_hallucinated"] for p in papers)
        logger.info(f"HaluEval loaded: {len(papers)} samples parsed, {n_hall} hallucinated ({n_hall/len(papers):.1%})")
    else:
        logger.info(f"Generating synthetic corpus of {args.n_papers} papers (rate: {args.hallucination_rate:.1%})...")
        papers = generate_corpus(n_papers=args.n_papers, hallucination_rate=args.hallucination_rate, seed=args.seed)
        n_hall = sum(p["is_hallucinated"] for p in papers)
        logger.info(f"Corpus generated: {len(papers)} papers total, {n_hall} hallucinated ({n_hall/len(papers):.1%})")
        
    logger.info("Training classifiers and performing 5-fold cross-validation...")
    eval_res = train_and_evaluate(
        papers=papers,
        mode=args.mode,
        dataset_path=args.dataset_csv,
        live_citation_check=args.live_citation_check,
        seed=args.seed
    )
    
    logger.info("--- 5-Fold Cross-Validation ROC-AUC Results ---")
    for name, res in eval_res["cv_results"].items():
        logger.info(f"  {name:20}: AUC = {res['auc_mean']:.3f} ± {res['auc_std']:.3f}")
        
    logger.info("--- Final Random Forest Performance (Test Split) ---")
    logger.info(f"  AUC:               {eval_res['final_auc']:.3f}")
    logger.info(f"  Average Precision: {eval_res['final_ap']:.3f}")
    logger.info(f"  Weighted F1-score: {eval_res['final_f1']:.3f}")
    logger.info(f"  Accuracy:          {eval_res['final_acc']:.3f}")
    
    results_out = {
        "mode": args.mode,
        "cv_results": eval_res["cv_results"],
        "final_model": {
            "auc": eval_res["final_auc"],
            "average_precision": eval_res["final_ap"],
            "f1_weighted": eval_res["final_f1"],
            "accuracy": eval_res["final_acc"]
        },
        "feature_importances": dict(zip(eval_res["feature_names"], eval_res["importances"].tolist())),
        "top_features": [eval_res["feature_names"][i] for i in np.argsort(eval_res["importances"])[::-1][:5]]
    }
    
    if args.mode in ("synthetic", "online") and papers is not None:
        y = eval_res["y"]
        citation_features = [analyse_citations(p, live=args.live_citation_check) for p in papers]
        stat_features = [analyse_statistics(p) for p in papers]
        confidence_scores = np.array([calculate_confidence_score(p) for p in papers])
        
        hallucination_scores = (
            np.array([cf["citation_suspicion_score"] for cf in citation_features]) * 0.45 +
            np.array([sf["stat_suspicion_score"] for sf in stat_features]) * 0.35 +
            confidence_scores * 0.20
        )
        
        from sklearn.metrics import roc_auc_score
        layer_aucs = {
            "Citation forensics": float(roc_auc_score(y, [cf["citation_suspicion_score"] for cf in citation_features])),
            "Statistical forensics": float(roc_auc_score(y, [sf["stat_suspicion_score"] for sf in stat_features])),
            "Confidence mismatch": float(roc_auc_score(y, confidence_scores)),
            "Composite score": float(roc_auc_score(y, hallucination_scores)),
            "Random Forest (ML)": float(roc_auc_score(y, eval_res["rf_final"].predict_proba(eval_res["X_noise"])[:, 1]))
        }
        results_out["layer_aucs"] = layer_aucs
        results_out["corpus"] = {
            "n_papers": len(papers),
            "n_hallucinated": int(n_hall)
        }
        
        logger.info("--- Layer-by-Layer AUC (Full Dataset) ---")
        for layer, auc in layer_aucs.items():
            logger.info(f"  {layer:25}: AUC = {auc:.3f}")
            
    logger.info(f"Saving results JSON to: {args.results_file}")
    save_results(results_out, args.results_file)
    
    if not args.no_plots:
        logger.info(f"Generating and saving diagnostic figures in directory: {args.output_dir} ...")
        generate_figures(papers, eval_res, args.output_dir, mode=args.mode)
        logger.info("Diagnostic figures saved successfully.")
        
    logger.info("=" * 60)
    logger.info("P9 EXPERIMENT COMPLETE  ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
