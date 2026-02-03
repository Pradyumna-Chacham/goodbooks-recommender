# 📚 Goodbooks Recommendation System  
*Adaptive Book Recommendation with Collaborative Filtering and Reinforcement Learning*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
[![Tests](https://github.com/Pradyumna-Chacham/goodbooks-rec/actions/workflows/tests.yml/badge.svg)](https://github.com/Pradyumna-Chacham/goodbooks-rec/actions/workflows/tests.yml)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-64%25-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

![Code%20Style](https://img.shields.io/badge/Code%20Style-Black-black)
![Imports](https://img.shields.io/badge/Imports-isort-blue)
![Linting](https://img.shields.io/badge/Linting-Pylint-yellowgreen)
![Dataset](https://img.shields.io/badge/Dataset-Goodbooks--10k-purple)


---

## 🌟 Project Overview

The **Goodbooks Recommendation System** is an end-to-end, research-driven recommender system built on the **Goodbooks-10k dataset**.  
It explores and compares **classical collaborative filtering**, **content-based methods**, **latent factor models**, and **reinforcement learning–based reranking**, with an emphasis on **realistic deployment constraints**.

Rather than focusing purely on offline metrics, this project delivers an **interactive Streamlit demo** that allows users to:
* Explore multiple recommendation strategies.
* Compare model behavior side-by-side.
* Understand trade-offs between simplicity, performance, and scalability.

The system culminates in a **hybrid CF + RL reranking architecture**, inspired by modern industrial recommender pipelines.

---

## 🏗️ System Architecture

```text
Offline Pipeline
├── Dataset cleaning & filtering
├── Collaborative filtering models
├── Latent factor embeddings
├── Reinforcement learning training
└── Artifact serialization

Online Pipeline
├── Streamlit UI
├── Cached artifact loading
├── Multi-model inference layer
└── Interactive visualization

```

---

## 🧠 Models Implemented

| Model | Technique | Best For |
| --- | --- | --- |
| **Popularity Baseline** | Global frequency ranking | Global trends / Baseline |
| **Content-Based** | TF-IDF (Titles, Authors, Tags) | Cold-start / Explainability |
| **Item-Based CF** | Cosine similarity on ratings | Dense book data / Candidate generation |
| **User-Based CF** | Mean-centered user similarity | High personalization |
| **SVD Hybrid** | Matrix Factorization + PCA Tags | Handling sparsity |
| **RL (Q-Learning)** | Deep Q-Learning on embeddings | Dynamic ranking preferences |
| **Hybrid CF + RL** ⭐ | **CF Retrieval + RL Reranking** | State-of-the-art performance |

### The Hybrid CF + RL Reranker

This is the flagship model. It mirrors real-world architectures used in large-scale systems:

1. **Item-CF** retrieves a high-quality candidate set.
2. **RL Agent** reranks candidates based on learned reward signals.
3. **Final Scores** combine CF similarity and RL output for optimized ranking.

---

## 📈 Evaluation

All models are evaluated under a uniform, leakage-free protocol.

| Model | HR@5 (Hit Rate) | NDCG@5 |
| --- | --- | --- |
| **Hybrid CF + RL** | **0.4260** | **0.2819** |
| Item-Based CF | 0.4012 | 0.2543 |
| SVD Hybrid | 0.3845 | 0.2310 |

---

## 👤 Personas & User Simulation

To demonstrate personalization without requiring user authentication, the app includes representative personas mapped to real user IDs:

* **Fantasy Fan**
* **Sci-Fi Enthusiast**
* **Mystery / Thriller Reader**
* **Romance Reader**
* **History Reader**
* **YA / Teen Reader**
* **Classics Lover**

---

## 📁 Project Structure

* `apps/`: Streamlit application and multi-page UI.
* `src/goodbooks_rec/`: Core recommendation library (logic, loading, personas).
* `scripts/`: Offline training, preprocessing, and data fetching.
* `data/`: Local dataset storage (Git ignored).
* `models/`: Trained model artifacts (Git ignored).
* `tests/`: Comprehensive unit and integration tests.

---

## 🚀 Getting Started

1. **Installation:** Refer to [INSTALL.md](INSTALL.md) for environment setup.
2. **Development:** Refer to [SETUP.md](SETUP.md) for architecture details.

**Quick Start:**

```bash
bash scripts/download_data.sh
streamlit run apps/streamlit_app.py

```

---

## 🧪 Testing

The project includes 26 tests covering core logic and integration.

```bash
pytest --cov=goodbooks_rec --cov-report=term-missing

```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📖 Citation

If you use this project in academic work, please cite:

```bibtex
@software{goodbooksrec2026,
  title = {Goodbooks Recommendation System},
  author = {Pradyumna Chacham},
  year = {2026},
  url = {[https://github.com/](https://github.com/)<your-username>/goodbooks-rec},
  note = {Adaptive recommender system with collaborative filtering and reinforcement learning}
}

```
