# Contributing to Academic Hallucination Detector

We welcome contributions to improve the multi-layer hallucination detection pipeline.

## Development Workflow

1. Clone the repository and install the development dependencies:
   ```bash
   pip install -e .[dev]
   ```

2. Format and lint your code using `ruff`:
   ```bash
   ruff check .
   ruff format .
   ```

3. Run the unit tests to ensure everything is correct:
   ```bash
   python -m pytest
   ```

4. Run the validation experiments to verify that diagnostic outputs remain stable:
   ```bash
   python experiments/p9_hallucination_validation.py --no-plots
   ```

## Pull Request Guidelines

- Ensure all existing unit tests pass.
- Write unit tests for any new features or forensic heuristics.
- Maintain clear documentation and clean docstrings in all python modules.
