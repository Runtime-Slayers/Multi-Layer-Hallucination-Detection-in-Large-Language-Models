# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-06-25

### Added
- Created the modular `hallucination_detector` package (`corpus.py`, `citation.py`, `statistics.py`, `confidence.py`, `classifier.py`, `utils.py`).
- Added robust validation experiment runner `experiments/p9_hallucination_validation.py` with parameter configuration and UTF-8 console output.
- Developed comprehensive test suite `tests/test_detector.py` covering all packages.
- Added project configurations (`pyproject.toml`, `requirements.txt`, `conftest.py`, `.gitignore`).
- Documented full mathematical theory in `docs/THEORY.md` and updated `README.md` with equations.
- Configured GitHub Actions workflow `.github/workflows/ci.yml`.
