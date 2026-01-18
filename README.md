# 📚 Adaptive Book Recommendation System using Collaborative Filtering and Reinforcement Learning  
### ALDA (CSC 522) — Final Project, Fall 2025  
**Team P05**  
Pradyumna Chacham · Sai Mahathi Suryadevara · Praneetha Koppala  
NC State University — Department of Computer Science  

---

## 📖 Project Overview

This project explores the design of an **adaptive offline book recommender system** using the **Goodbooks-10k** dataset.  
We implement and compare three major families of recommendation algorithms:

1. **Classical Baselines**  
   - Popularity  
   - Content-Based Filtering (CBF)  
   - Item-Based Collaborative Filtering (Item-CF)  
   - User-Based Collaborative Filtering (User-CF)

2. **Latent Factor Model**  
   - Hybrid **Matrix Factorization + PCA** model (100-d MF + 50-d PCA book genres)

3. **Reinforcement Learning Models**  
   - *Pure RL* (Q-learning on hybrid embeddings)  
   - *CF+RL Hybrid Reranker* (Item-CF generates candidates, RL reranks them)

Our goal is to evaluate whether reinforcement learning can improve top-K recommendation performance in an offline environment and identify which components benefit from RL and which do not.

## 🗂 Repository Structure

```.
├── book_genres.py # Tag / genre extraction and cleaning
├── build_dataset.py # Dataset preprocessing + dense subset creation
├── clean_tags.py # Tag normalization and filtering
├── train_baselines.py # Classical baselines + Hybrid SVD model
├── train_rl_hybrid.py # Pure RL (Q-learning) training
├── train_rl_reranker.py # CF+RL reranking model (best-performing)
├── visualization.py # Generates pipeline and metric plots
├── requirements.txt
├── report/ # Final LaTeX report and figures
└── README.md
```

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/pchacha2_ncstate/ncsu-engr-ALDA-F25-Project-P05.git
cd ncsu-engr-ALDA-F25-Project-P05 
```
Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install requirements:
```
pip install -r requirements.txt
```
📂 Dataset

Get the dataset from the following link: https://github.com/zygmuntz/goodbooks-10k/releases

Extract the data files to "data" folder, and then run the following commands



▶️ Running the Project
1. Preprocess Data
```
python build_dataset.py
python clean_tags.py
```
3. Train Classical Baselines
```
python train_baselines.py
```

4. Train the RL Model
```
python train_rl_hybrid.py
```
5. Train CF+RL Reranker
```
python train_rl_reranker.py
```

This project includes a full interactive terminal tool for evaluating recommendations:

 Run:
 ```
python inference.py
```
Refer to the sample output in Output.md file
