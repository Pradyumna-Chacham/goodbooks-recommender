```markdown
# 🧰 Setup Guide — Goodbooks Recommendation System

This guide explains how to **set up the project for development**, understand the repository structure, and work effectively with models, data, tests, and the Streamlit app.

Use this after completing the steps in **INSTALL.md**.

---

## 🎯 Purpose of This Guide

This document focuses on:
* Project structure and responsibilities
* Development workflow
* Dataset and artifact handling
* Testing and experimentation
* Running and modifying the Streamlit app

---

## 📁 Repository Structure

```text
.
├── apps/                   # Streamlit application
│   ├── streamlit_app.py    # App entrypoint
│   └── pages/              # Multi-page Streamlit UI
│
├── src/goodbooks_rec/      # Core recommendation library
│   ├── io.py               # Artifact loading
│   ├── recommend.py        # Simple recommenders
│   ├── multimodel.py       # Multi-model + hybrid logic
│   ├── personas.py         # Demo personas
│   ├── ui.py               # UI helpers (non-Streamlit)
│   └── config.py           # Global config/constants
│
├── scripts/                # Offline preprocessing & training
│   ├── build_dataset.py
│   ├── clean_tags.py
│   ├── train_baselines.py
│   ├── train_rl_reranker.py
│   └── download_data.sh
│
├── models/                 # Trained model artifacts (ignored by git)
│
├── data/                   # Goodbooks dataset (ignored by git)
│
├── tests/                  # Unit + integration tests
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── docs/                   # Documentation and report
│
├── README.md
├── INSTALL.md
├── SETUP.md
├── CHANGELOG.md
└── pyproject.toml

```

---

## 🧠 Development Philosophy

This project follows a clean separation of concerns:

* **Core logic** lives in `src/goodbooks_rec/`
* **Streamlit UI** is a thin wrapper
* **Heavy computation** is done offline
* **Artifacts** are precomputed and loaded at runtime
* **Tests** focus on logic, not UI

This design keeps the system reproducible, testable, and deployable within memory constraints.

---

## 📊 Dataset Handling

### Dataset Location

Dataset files live in `./data`. This directory is **ignored by git**.
Files are downloaded via:

```bash
bash scripts/download_data.sh

```

### Expected Files

```text
data/
├── books.csv
├── ratings.csv
├── tags.csv
├── book_tags.csv
└── to_read.csv

```

### Design Choice

The dataset is public and large, so it is not versioned in Git. It is downloaded locally and referenced consistently by internal loaders.

---

## 🏗️ Model Artifacts

### Artifact Location

The `models/` directory is **ignored by git**. Artifacts are generated offline via scripts:

* `item_topk_k100.npz`
* `user_topk_k100.npz`
* `svd_hybrid.pkl`
* `hybrid_reranker.pkl`
* `rl_cfhard_fast.pth`

### Loading Logic

All artifact loading is centralized in `goodbooks_rec/io.py`. This ensures one source of truth and allows for easy Streamlit caching support.

---

## 🧪 Testing Setup

### Test Structure

```text
tests/
├── unit/          # Fast, isolated tests
├── integration/   # Cross-module behavior
├── fixtures/      # Mini datasets & models
└── helpers/       # Test utilities

```

### Commands

* **Run Tests:** `pytest`
* **Coverage:** `pytest --cov=goodbooks_rec --cov-report=term-missing`

### Design Notes

* Mini fixtures allow tests to run quickly without loading the full 10k dataset.
* Streamlit UI code is intentionally excluded from unit testing.

---

## 🖥️ Streamlit App Setup

### App Entry Point

```bash
streamlit run apps/streamlit_app.py

```

### Page Structure

Each page under `apps/pages/` focuses on a single model or comparison and calls shared logic from `goodbooks_rec` to avoid duplication.

### State Management

* Pagination and UI state use `st.session_state`.
* Artifacts are cached with `st.cache_resource`.

---

## 🔁 Development Workflow

1. **Modify** core logic in `src/goodbooks_rec/`.
2. **Add/Update** tests in `tests/`.
3. **Run** `pytest` to ensure no regressions.
4. **Run** the Streamlit app locally to verify UI changes.
5. **Commit** changes.

---

## ☁️ Streamlit Community Cloud Setup Notes

* Ensure `requirements.txt` is complete.
* Artifacts must fit within memory constraints (approx. 1GB).
* Cache heavy resources aggressively using `@st.cache_resource`.

---

## 🧭 Where to Go Next

* **INSTALL.md** → Installation instructions
* **README.md** → Project overview and results
* **docs/** → Full technical report

---

✅ **Setup Complete**
You’re ready to extend, refactor, or deploy! 🚀

