# Changelog

All notable changes to this project will be documented in this file.

The format is based on **Keep a Changelog**, and this project aims to follow **Semantic Versioning**.

---

## [Unreleased]

### Added
- Dataset download script (`scripts/download_data.sh`) to fetch the Goodbooks dataset locally
- Installation guide (`INSTALL.md`) with step-by-step setup instructions
- Project setup guide (`SETUP.md`) describing architecture and development workflow
- GitHub issue templates for bug reports and feature requests
- Pull request template for consistent contributions

### Changed
- Repository now ignores dataset files (`data/`) and trained artifacts (`models/`)
- Improved documentation structure under `docs/`
- Clarified Streamlit deployment assumptions in documentation

### Fixed
- Minor documentation inconsistencies across README and setup guides

---

## [0.1.0] – 2026-02-02

### Added
- Initial public release of the Goodbooks Recommendation System
- Multi-model recommendation framework implemented in `goodbooks_rec`:
  - Popularity baseline
  - Content-Based Filtering (TF-IDF)
  - Item-Based Collaborative Filtering
  - User-Based Collaborative Filtering
  - Hybrid SVD latent factor model
  - Reinforcement Learning (Q-learning) recommender
  - Hybrid CF + RL reranking model
- Streamlit multi-page demo application:
  - Individual pages for each recommendation strategy
  - Model comparison and evaluation view
  - Persona-based user simulation
- Offline preprocessing and training scripts:
  - Dataset cleaning and filtering
  - Top-K similarity matrix generation
  - Baseline and RL model training
- Centralized artifact loading with caching support
- Memory-optimized deployment design targeting Streamlit Community Cloud
- Comprehensive test suite:
  - Unit tests for core logic
  - Integration tests for seen-item filtering
  - Mini fixtures for fast and reproducible testing
- GitHub Actions workflow to run tests on push and pull requests

### Design Decisions
- Dataset and trained model artifacts are not versioned
- Heavy computation is performed offline; runtime inference is lightweight
- Streamlit UI acts as a thin layer over reusable recommendation logic
- Reinforcement learning is used strictly as a reranker, not a standalone retriever

---

## Versioning Notes

- **MAJOR** version increments indicate breaking API or architectural changes
- **MINOR** version increments indicate new models or user-facing features
- **PATCH** versions indicate bug fixes, refactors, or documentation updates
