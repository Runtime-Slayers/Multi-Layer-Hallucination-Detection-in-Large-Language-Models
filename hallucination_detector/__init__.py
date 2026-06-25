"""
Academic Hallucination Detector package.
Detects AI hallucinations in academic writing using multi-layer forensics.
"""

__version__ = "0.1.0"

from .corpus import (
    generate_corpus,
    REAL_JOURNALS,
    FAKE_JOURNALS,
    LAST_NAMES,
    CONFIDENCE_PHRASES_HIGH,
    CONFIDENCE_PHRASES_MED
)
from .citation import analyse_citations
from .statistics import analyse_statistics
from .confidence import calculate_confidence_score, calculate_confidence_scores
from .classifier import build_feature_matrix, train_and_evaluate
from .utils import save_results, generate_figures

__all__ = [
    "generate_corpus",
    "REAL_JOURNALS",
    "FAKE_JOURNALS",
    "LAST_NAMES",
    "CONFIDENCE_PHRASES_HIGH",
    "CONFIDENCE_PHRASES_MED",
    "analyse_citations",
    "analyse_statistics",
    "calculate_confidence_score",
    "calculate_confidence_scores",
    "build_feature_matrix",
    "train_and_evaluate",
    "save_results",
    "generate_figures"
]
