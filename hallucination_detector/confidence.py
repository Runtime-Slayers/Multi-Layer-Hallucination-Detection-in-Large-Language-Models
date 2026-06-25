"""
Confidence-accuracy mismatch module for the academic hallucination detector.
Analyzes linguistic overconfidence markers relative to scientific hedging.
"""


def calculate_confidence_score(paper):
    """
    Calculate confidence score for a single paper based on high confidence phrase
    fraction and raw confidence phrase count.
    """
    high_conf_frac = paper.get("high_conf_frac", 0.0)
    conf_phrase_count = paper.get("confidence_phrase_count", 0)
    
    score = high_conf_frac * 0.6 + (conf_phrase_count / 10.0) * 0.4
    return float(max(0.0, min(1.0, score)))


def calculate_confidence_scores(papers):
    """Calculate confidence scores for a list of papers."""
    return [calculate_confidence_score(p) for p in papers]
