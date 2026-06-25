"""
Corpus generation module for the academic hallucination detector.
Generates a synthetic corpus of papers with realistic citation, statistical, and linguistic markers.
"""

import numpy as np

REAL_JOURNALS = [
    "Nature", "Science", "JAMA", "Lancet", "PNAS", "Cell",
    "Nature Neuroscience", "Psychological Science", "PLOS ONE",
    "Journal of Educational Psychology", "Cognition", "NeuroImage",
    "British Medical Journal", "New England Journal of Medicine"
]

FAKE_JOURNALS = [
    "Journal of Advanced Cognitive Neuroscience",
    "International Review of Learning Sciences",
    "European Mind & Brain Research",
    "Global Journal of Educational Intelligence",
    "Frontiers in Neural Computing Research",
    "Advanced Studies in Psychological Technology"
]

LAST_NAMES = [
    "Johnson", "Williams", "Smith", "Brown", "Davis", "Miller", "Wilson",
    "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Garcia", "Martinez", "Lee", "Thompson", "Clark"
]

YEARS_VALID = list(range(2015, 2026))
YEARS_FUTURE = list(range(2026, 2035))

CONFIDENCE_PHRASES_HIGH = [
    "It is well established that", "As demonstrated conclusively by",
    "The landmark study by", "All researchers agree that",
    "Definitively proven by", "The seminal work of"
]

CONFIDENCE_PHRASES_MED = [
    "Studies suggest that", "Evidence indicates", "Research shows",
    "Findings suggest", "Results demonstrate", "Analysis reveals"
]


def gen_author(hallucinated=False, rng=None):
    """Generate a single author name, occasionally format differently for hallucinations."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = rng.choice(LAST_NAMES)
    if hallucinated and rng.random() < 0.12:
        return f"X. {n}"
    elif not hallucinated and rng.random() < 0.02:
        return f"X. {n}"
    return f"{n[0]}. {n}"


def gen_citation(hallucinated=False, rng=None):
    """Generate a mock citation with potential hallucination markers."""
    if rng is None:
        rng = np.random.default_rng()
    
    n_authors = rng.integers(1, 5)
    authors = ", ".join([gen_author(hallucinated, rng) for _ in range(n_authors)])
    r = rng.random()
    
    if hallucinated and r < 0.15:
        year = rng.choice(YEARS_FUTURE)
    elif not hallucinated and r < 0.02:
        year = rng.choice(YEARS_FUTURE)
    elif hallucinated and r < 0.30:
        year = rng.integers(1950, 2000)
    else:
        year = rng.choice(YEARS_VALID)
        
    if hallucinated:
        journal = rng.choice(FAKE_JOURNALS if rng.random() < 0.45 else REAL_JOURNALS)
    else:
        journal = rng.choice(FAKE_JOURNALS if rng.random() < 0.20 else REAL_JOURNALS)
        
    vol = rng.integers(1, 50)
    page = rng.integers(1, 999)
    
    return {
        "authors": authors,
        "year": int(year),
        "journal": journal,
        "volume": int(vol),
        "page": int(page),
        "is_hallucinated": hallucinated
    }


def gen_statistic(hallucinated=False, rng=None):
    """Generate mock statistical claims (percentage, correlation, p-value, etc.)."""
    if rng is None:
        rng = np.random.default_rng()
        
    if hallucinated:
        actually_suspicious = rng.random() < 0.35
    else:
        actually_suspicious = rng.random() < 0.15

    if actually_suspicious:
        stat_type = rng.choice(["pct", "corr", "pval", "effect"])
        if stat_type == "pct":
            val = rng.choice([float(v) for v in [12, 23, 34, 45, 56, 67, 78, 89, 43, 57]])
            return {"type": "percentage", "value": val, "is_hallucinated": True,
                    "suspicion": "round_number"}
        elif stat_type == "corr":
            val = round(float(rng.uniform(0.85, 0.99)), 2)
            return {"type": "correlation", "value": val, "is_hallucinated": True,
                    "suspicion": "implausibly_high"}
        elif stat_type == "pval":
            val = float(rng.choice([0.001, 0.0001, 0.00001, 0.000001]))
            return {"type": "p_value", "value": val, "is_hallucinated": True,
                    "suspicion": "suspiciously_small"}
        else:
            val = round(float(rng.uniform(1.5, 3.5)), 1)
            return {"type": "effect_size", "value": val, "is_hallucinated": True,
                    "suspicion": "large_effect"}
    else:
        stat_type = rng.choice(["pct", "corr", "pval", "effect"])
        if stat_type == "pct":
            val = round(float(rng.uniform(10, 85) + rng.normal() * 5), 1)
            return {"type": "percentage", "value": val, "is_hallucinated": False, "suspicion": "none"}
        elif stat_type == "corr":
            val = round(float(rng.uniform(0.2, 0.75)), 2)
            return {"type": "correlation", "value": val, "is_hallucinated": False, "suspicion": "none"}
        elif stat_type == "pval":
            val = round(float(rng.uniform(0.01, 0.05)), 3)
            return {"type": "p_value", "value": val, "is_hallucinated": False, "suspicion": "none"}
        else:
            val = round(float(rng.uniform(0.2, 1.2)), 2)
            return {"type": "effect_size", "value": val, "is_hallucinated": False, "suspicion": "none"}


def generate_corpus(n_papers=600, hallucination_rate=0.45, seed=42):
    """Generate a synthetic corpus of papers with simulated forensic characteristics."""
    rng = np.random.default_rng(seed)
    papers = []
    
    for pid in range(n_papers):
        is_hallucinated = rng.random() < hallucination_rate
        n_citations = int(rng.integers(8, 25))
        n_stats = int(rng.integers(3, 10))

        if is_hallucinated:
            n_fake = max(1, int(n_citations * rng.uniform(0.3, 0.8)))
            citations = [gen_citation(hallucinated=(i < n_fake), rng=rng) for i in range(n_citations)]
            rng.shuffle(citations)
        else:
            citations = [gen_citation(hallucinated=False, rng=rng) for _ in range(n_citations)]

        statistics = [gen_statistic(hallucinated=is_hallucinated, rng=rng) for _ in range(n_stats)]

        if is_hallucinated:
            confidence_phrase_count = int(rng.integers(0, 6))
            high_conf_frac = float(np.clip(rng.normal(0.55, 0.22), 0.0, 1.0))
        else:
            confidence_phrase_count = int(rng.integers(0, 5))
            high_conf_frac = float(np.clip(rng.normal(0.42, 0.22), 0.0, 1.0))

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
        
    return papers
