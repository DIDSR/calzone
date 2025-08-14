# Contributing to Calzone

Thank you for your interest in contributing to **Calzone**! We welcome any improvements that help ensure the project remains reliable, maintainable, and of high-quality.

---

##  How to Contribute

1. Fork the repository and create working branches from `main`.  
2. Open a pull request (PR) for your changes, ideally linked to an issue that describes the feature or bug.  
3. Ensure all CI checks pass (tests, linting, formatting) before requesting review.  

---

##  Project Standards and Guidelines

### 1. Statistical Tests & Metrics

- Every new statistical test or metric must include a **unit test** under the `tests/` directory.  
- Whenever possible, unit tests should include:
  - **Comparisons with results from existing packages**, or  
  - **Analytical or known-case tests** to validate correctness.

### 2. New Metrics: Confidence Intervals

- Any newly introduced metric must be delivered with a method to compute **confidence intervals**.  
- Please document:
  - The **methodology** (e.g., parametric bootstrap, analytical derivation), and  
  - Any **assumptions or parameters** required.

### 3. Code Formatting

- Before committing, run **Black** to format your Python code consistently:
  ```bash
  black .
