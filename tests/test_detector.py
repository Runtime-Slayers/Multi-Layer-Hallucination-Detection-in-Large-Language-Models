"""
Unit tests for the academic hallucination detector package.
"""

import pytest
import numpy as np

from hallucination_detector import (
    generate_corpus,
    analyse_citations,
    analyse_statistics,
    calculate_confidence_score,
    calculate_confidence_scores,
    build_feature_matrix,
    train_and_evaluate
)


def test_generate_corpus():
    """Test synthetic corpus generation parameters and structure."""
    n_papers = 30
    rate = 0.5
    papers = generate_corpus(n_papers=n_papers, hallucination_rate=rate, seed=42)
    
    assert len(papers) == n_papers
    for p in papers:
        assert "id" in p
        assert "is_hallucinated" in p
        assert "citations" in p
        assert "statistics" in p
        assert "confidence_phrase_count" in p
        assert "high_conf_frac" in p
        assert "n_citations" in p
        assert "n_stats" in p
        assert len(p["citations"]) == p["n_citations"]
        assert len(p["statistics"]) == p["n_stats"]


def test_analyse_citations():
    """Test citation verification forensics scoring."""
    # Scenario 1: Low suspicion (all real journals, valid past years)
    paper_clean = {
        "citations": [
            {"journal": "Nature", "year": 2020, "authors": "A. Smith"},
            {"journal": "Science", "year": 2022, "authors": "B. Jones"}
        ]
    }
    res_clean = analyse_citations(paper_clean)
    assert res_clean["n_fake_journals"] == 0
    assert res_clean["n_future_years"] == 0
    assert res_clean["frac_fake_journals"] == 0.0
    assert res_clean["frac_future_years"] == 0.0
    assert res_clean["citation_suspicion_score"] == pytest.approx(0.0)

    # Scenario 2: High suspicion (fake journals, future years, ghost markers)
    paper_dirty = {
        "citations": [
            {"journal": "Journal of Advanced Cognitive Neuroscience", "year": 2030, "authors": "X. Johnson"},
            {"journal": "Science", "year": 2022, "authors": "B. Jones"}
        ]
    }
    res_dirty = analyse_citations(paper_dirty)
    assert res_dirty["n_fake_journals"] == 1
    assert res_dirty["n_future_years"] == 1
    assert res_dirty["frac_fake_journals"] == 0.5
    assert res_dirty["frac_future_years"] == 0.5
    assert res_dirty["n_ghost_markers"] == 1
    # score calculation: 0.5*0.5 + 0.5*0.3 + (1-0.5)*0.1 + 0.5*0.1 = 0.25 + 0.15 + 0.05 + 0.05 = 0.50
    assert res_dirty["citation_suspicion_score"] == pytest.approx(0.50)


def test_analyse_statistics():
    """Test statistical forensics scoring."""
    # Scenario 1: Clean stats
    paper_clean = {
        "statistics": [
            {"type": "percentage", "value": 45.3, "suspicion": "none"},
            {"type": "correlation", "value": 0.42, "suspicion": "none"}
        ]
    }
    res_clean = analyse_statistics(paper_clean)
    assert res_clean["n_suspicious_stats"] == 0
    assert res_clean["round_number_frac"] == 0.0
    assert res_clean["stat_suspicion_score"] == pytest.approx(0.0)

    # Scenario 2: Suspicious stats (round percentage, implausibly high correlation)
    paper_dirty = {
        "statistics": [
            {"type": "percentage", "value": 56.0, "suspicion": "round_number"},
            {"type": "correlation", "value": 0.98, "suspicion": "implausibly_high"}
        ]
    }
    res_dirty = analyse_statistics(paper_dirty)
    assert res_dirty["n_suspicious_stats"] == 2
    assert res_dirty["round_number_frac"] == 1.0
    # score calculation: (2/2)*0.6 + 1.0*0.4 = 1.0
    assert res_dirty["stat_suspicion_score"] == pytest.approx(1.0)


def test_confidence_scores():
    """Test confidence language score calculations."""
    paper = {
        "high_conf_frac": 0.5,
        "confidence_phrase_count": 5
    }
    # score: 0.5 * 0.6 + (5/10) * 0.4 = 0.3 + 0.2 = 0.5
    assert calculate_confidence_score(paper) == pytest.approx(0.5)
    
    papers = [paper, {"high_conf_frac": 0.0, "confidence_phrase_count": 0}]
    scores = calculate_confidence_scores(papers)
    assert len(scores) == 2
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(0.0)


def test_feature_matrix_and_training():
    """Test building feature matrix and training cross-validated models."""
    papers = generate_corpus(n_papers=20, hallucination_rate=0.5, seed=42)
    df = build_feature_matrix(papers)
    
    assert len(df) == 20
    assert "citation_suspicion" in df.columns
    assert "stat_suspicion" in df.columns
    assert "confidence_score" in df.columns
    
    # Train and evaluate classifier
    eval_res = train_and_evaluate(papers, seed=42)
    assert "cv_results" in eval_res
    assert "Logistic Regression" in eval_res["cv_results"]
    assert "Random Forest" in eval_res["cv_results"]
    assert "Gradient Boosting" in eval_res["cv_results"]
    
    assert "rf_final" in eval_res
    assert eval_res["final_auc"] >= 0.0
    assert len(eval_res["importances"]) == len(eval_res["feature_names"])
