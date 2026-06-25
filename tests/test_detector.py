"""
Unit tests for the academic hallucination detector package.
"""

import pytest
import os
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
from hallucination_detector.citation import CitationVerifier
from hallucination_detector.corpus import load_halueval_as_papers, download_halueval_dataset


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
    res_clean = analyse_citations(paper_clean, live=False)
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
    res_dirty = analyse_citations(paper_dirty, live=False)
    assert res_dirty["n_fake_journals"] == 1
    assert res_dirty["n_future_years"] == 1
    assert res_dirty["frac_fake_journals"] == 0.5
    assert res_dirty["frac_future_years"] == 0.5
    assert res_dirty["n_ghost_markers"] == 1
    # score calculation: 0.5*0.5 + 0.5*0.3 + (1-0.5)*0.1 + 0.5*0.1 = 0.25 + 0.15 + 0.05 + 0.05 = 0.50
    assert res_dirty["citation_suspicion_score"] == pytest.approx(0.50)


def test_citation_verifier_live_and_offline():
    """Test CitationVerifier class behavior in offline mode and API mocking."""
    verifier = CitationVerifier()
    
    # Offline checks
    res_offline = verifier.verify_citation("A. Smith (2020). Title. Nature.", live=False)
    assert res_offline["verdict"] == "EXISTS"
    assert res_offline["confidence"] == 0.95
    
    res_offline_bad = verifier.verify_citation("X. Johnson (2030). Title. Journal of Advanced Cognitive Neuroscience.", live=False)
    assert res_offline_bad["verdict"] == "NOT_FOUND"
    assert res_offline_bad["confidence"] == 0.85

    # Mocked API checks (handling network unavailability gracefully)
    parsed = verifier._parse_citation("A. Smith (2020). Title. Nature.")
    assert parsed["authors"] == "A. Smith"
    assert parsed["year"] == 2020
    assert parsed["title"] == "Title"
    
    cr = verifier._check_crossref({"title": ""})
    assert not cr["found"]
    
    ss = verifier._check_semantic_scholar({"title": ""})
    assert not ss["found"]


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


def test_csv_mode_training():
    """Test loading and training on local CSV dataset."""
    csv_path = "data/P9_llm_detection_dataset.csv"
    if os.path.exists(csv_path):
        res = train_and_evaluate(mode="csv", dataset_path=csv_path, seed=42)
        assert "cv_results" in res
        assert "Random Forest" in res["cv_results"]
        assert "Logistic Regression" in res["cv_results"]
        assert len(res["feature_names"]) == 3
        assert "perplexity" in res["feature_names"]
    else:
        pytest.skip("Local P9_llm_detection_dataset.csv not found, skipping CSV training test.")


def test_halueval_loading():
    """Test loading HaluEval dataset (handles offline fallback gracefully)."""
    cache_dir = "data/cache"
    papers = load_halueval_as_papers(cache_dir=cache_dir, subset="qa", limit=10)
    assert len(papers) > 0
    assert "is_hallucinated" in papers[0]
    assert "n_stats" in papers[0]
