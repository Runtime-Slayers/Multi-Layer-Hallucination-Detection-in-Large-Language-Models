"""
Machine learning classifier module for the academic hallucination detector.
Prepares features, runs cross-validation, and fits final models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report
)

from .citation import analyse_citations
from .statistics import analyse_statistics
from .confidence import calculate_confidence_score


def build_feature_matrix(papers, live=False):
    """Convert a list of paper dicts into a pandas DataFrame of features."""
    feature_rows = []
    for p in papers:
        cf = analyse_citations(p, live=live)
        sf = analyse_statistics(p)
        conf_score = calculate_confidence_score(p)
        
        row = {
            "n_citations": p.get("n_citations", 0),
            "n_stats": p.get("n_stats", 0),
            "frac_fake_journals": cf["frac_fake_journals"],
            "frac_future_years": cf["frac_future_years"],
            "n_ghost_markers": cf["n_ghost_markers"],
            "citation_suspicion": cf["citation_suspicion_score"],
            "n_suspicious_stats": sf["n_suspicious_stats"],
            "round_number_frac": sf["round_number_frac"],
            "stat_suspicion": sf["stat_suspicion_score"],
            "confidence_score": conf_score,
            "high_conf_frac": p.get("high_conf_frac", 0.0),
            "conf_phrase_count": p.get("confidence_phrase_count", 0),
        }
        feature_rows.append(row)
        
    df = pd.DataFrame(feature_rows)
    return df


def load_from_csv(file_path):
    """Load and parse dataset from a local CSV file."""
    df = pd.read_csv(file_path)
    return df


def train_and_evaluate(papers=None, mode="synthetic", dataset_path=None, live_citation_check=False, seed=42):
    """
    Build features, run cross-validation on multiple classifiers,
    and train the final Random Forest classifier on an 80/20 train/test split.
    Supports mode='synthetic', mode='csv', or mode='online'.
    """
    if mode == "csv":
        if dataset_path is None:
            raise ValueError("dataset_path must be provided when mode='csv'")
        df_raw = load_from_csv(dataset_path)
        if "hallucination_label" not in df_raw.columns:
            raise KeyError("CSV dataset must contain 'hallucination_label' column")
        df = df_raw.drop("hallucination_label", axis=1)
        y = df_raw["hallucination_label"].values
    elif mode == "online":
        if papers is None:
            raise ValueError("papers (processed dataset) must be provided when mode='online'")
        # online dataset is pre-processed into papers list
        df = build_feature_matrix(papers, live=live_citation_check)
        y = np.array([int(p.get("is_hallucinated", False)) for p in papers])
    else:
        if papers is None:
            raise ValueError("papers must be provided when mode='synthetic'")
        df = build_feature_matrix(papers, live=live_citation_check)
        y = np.array([int(p.get("is_hallucinated", False)) for p in papers])
        
    X = df.values
    
    # Add realistic measurement noise (imperfect feature extraction, OCR errors, etc.)
    rng_noise = np.random.RandomState(seed + 57)
    X_noise = X + rng_noise.standard_normal(X.shape) * 0.10 * (X.std(axis=0) + 1e-6)
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_noise)
    
    # Cross validation setup
    class_counts = np.bincount(y)
    min_class_size = int(class_counts.min()) if len(class_counts) > 1 else 0
    n_splits = min(5, min_class_size)
    
    models = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=seed),
    }
    
    cv_results = {}
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for name, model in models.items():
            X_in = X_sc if name == "Logistic Regression" else X_noise
            aucs = cross_val_score(model, X_in, y, cv=skf, scoring='roc_auc')
            cv_results[name] = {
                "auc_mean": float(aucs.mean()),
                "auc_std": float(aucs.std())
            }
    else:
        for name in models.keys():
            cv_results[name] = {
                "auc_mean": 0.0,
                "auc_std": 0.0
            }
        
    # Final Model: Train/Test Split (80/20)
    split = int(0.8 * len(y))
    rng_perm = np.random.default_rng(seed)
    idx = rng_perm.permutation(len(y))
    tr, te = idx[:split], idx[split:]
    
    rf_final = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf_final.fit(X_noise[tr], y[tr])
    
    y_prob = rf_final.predict_proba(X_noise[te])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    final_auc = float(roc_auc_score(y[te], y_prob))
    final_ap = float(average_precision_score(y[te], y_prob))
    
    report = classification_report(y[te], y_pred, output_dict=True)
    final_f1 = float(report["weighted avg"]["f1-score"])
    final_acc = float(report["accuracy"])
    
    feat_names = list(df.columns)
    importances = rf_final.feature_importances_
    
    return {
        "df_features": df,
        "X_noise": X_noise,
        "y": y,
        "test_indices": te,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "cv_results": cv_results,
        "rf_final": rf_final,
        "final_auc": final_auc,
        "final_ap": final_ap,
        "final_f1": final_f1,
        "final_acc": final_acc,
        "feature_names": feat_names,
        "importances": importances
    }

