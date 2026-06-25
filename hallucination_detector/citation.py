"""
Citation forensics module for the academic hallucination detector.
Analyzes citation lists for fake journals, future publication dates, and ghost authors.
"""

from .corpus import FAKE_JOURNALS, REAL_JOURNALS


def analyse_citations(paper):
    """
    Perform forensic analysis on a paper's citations list.
    Computes anomaly ratios and a composite citation suspicion score.
    """
    cites = paper.get("citations", [])
    total_cites = max(1, len(cites))
    
    n_fake_journals = sum(1 for c in cites if c.get("journal") in FAKE_JOURNALS)
    n_future_years = sum(1 for c in cites if c.get("year", 0) > 2025)
    n_plausible = sum(
        1 for c in cites 
        if c.get("journal") in REAL_JOURNALS and c.get("year", 0) <= 2025
    )
    n_ghost_marker = sum(
        1 for c in cites if str(c.get("authors", "")).startswith("X.")
    )
    
    frac_fake_j = n_fake_journals / total_cites
    frac_future = n_future_years / total_cites
    frac_real = n_plausible / total_cites
    
    citation_score = (
        frac_fake_j * 0.5 + 
        frac_future * 0.3 +
        (1.0 - frac_real) * 0.1 + 
        (n_ghost_marker / total_cites) * 0.1
    )
    
    return {
        "n_fake_journals": n_fake_journals,
        "n_future_years": n_future_years,
        "n_plausible": n_plausible,
        "n_ghost_markers": n_ghost_marker,
        "frac_fake_journals": frac_fake_j,
        "frac_future_years": frac_future,
        "citation_suspicion_score": float(max(0.0, min(1.0, citation_score)))
    }
