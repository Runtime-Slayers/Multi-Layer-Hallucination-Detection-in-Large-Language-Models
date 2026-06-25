"""
Statistical forensics module for the academic hallucination detector.
Analyzes statistical claims for psychological round numbers and implausible extremities.
"""


def analyse_statistics(paper):
    """
    Perform forensic analysis on a paper's statistical claims list.
    Checks for round numbers, implausibly high correlations, suspiciously small p-values,
    and extreme effect sizes.
    """
    stats_list = paper.get("statistics", [])
    n_total = max(1, len(stats_list))
    
    n_round = sum(1 for s in stats_list if s.get("suspicion") == "round_number")
    n_high_corr = sum(1 for s in stats_list if s.get("suspicion") == "implausibly_high")
    n_tiny_p = sum(1 for s in stats_list if s.get("suspicion") == "suspiciously_small")
    n_large_eff = sum(1 for s in stats_list if s.get("suspicion") == "large_effect")
    
    n_suspicious = n_round + n_high_corr + n_tiny_p + n_large_eff
    
    percentages = [s.get("value", 0.0) for s in stats_list if s.get("type") == "percentage"]
    if percentages:
        round_frac = sum(1 for v in percentages if v == round(v, 0)) / len(percentages)
    else:
        round_frac = 0.0
        
    stat_score = (n_suspicious / n_total) * 0.6 + round_frac * 0.4
    
    return {
        "n_suspicious_stats": n_suspicious,
        "round_number_frac": float(round_frac),
        "stat_suspicion_score": float(max(0.0, min(1.0, stat_score)))
    }
