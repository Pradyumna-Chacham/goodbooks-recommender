"""
hybrid_FINAL_CORRECT.py — Z-score + RandomSeed=42
"""

import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ============================================================
# GLOBAL RANDOM SEED FOR PERFECT REPRODUCIBILITY
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# ============================================================
# Load Data
# ============================================================
DATA = "new_data"
MODELS = "models"

print("=" * 60)
print("HYBRID: Item-CF + RL (CORRECT EVALUATION, Z-score, Seed=42)")
print("=" * 60)

print("\nLoading data...")
svd = pickle.load(open(f"{MODELS}/svd_hybrid.pkl", "rb"))
item_sim = pickle.load(open(f"{MODELS}/item_cf.pkl", "rb"))

user_embed = torch.tensor(svd["user_embed"], dtype=torch.float32)
item_embed = torch.tensor(svd["item_embed"], dtype=torch.float32)

user_to_idx = svd["user_to_idx"]
book_to_idx = svd["book_to_idx"]
idx_to_book = {v: k for k, v in book_to_idx.items()}

train_df = pickle.load(open(f"{DATA}/train_df.pkl", "rb"))
user_hist = train_df.groupby("user_id")["book_id"].apply(list).to_dict()

with open(f"{DATA}/eval_data.pkl", "rb") as f:
    eval_data = pickle.load(f)

print(f"  Users: {len(user_to_idx)}")
print(f"  Items: {len(book_to_idx)}")
print(f"  Eval users: {len(eval_data)}")

# ============================================================
# Precompute CF Vectors
# ============================================================
print("\nPrecomputing CF vectors...")
TOPK = 50
cf_vecs = np.zeros((len(book_to_idx), 50), dtype=np.float32)

for i in tqdm(range(len(book_to_idx)), desc="CF vectors"):
    sims = item_sim[i]
    top_idx = np.argsort(sims)[-TOPK:]
    top_embeds = item_embed[top_idx]
    cf_vecs[i] = top_embeds.mean(dim=0)[:50].numpy()

cf_vecs = torch.tensor(cf_vecs, dtype=torch.float32)


# ============================================================
# RL Model
# ============================================================
class QNet(nn.Module):
    def __init__(self, dim=150):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, sa):
        return self.net(sa)


device = torch.device("cpu")
qnet = QNet(dim=150).to(device)
qnet.load_state_dict(
    torch.load(f"{MODELS}/rl_cfhard_fast.pth", map_location=device, weights_only=True)
)
qnet.eval()

user_embed = user_embed.to(device)
item_embed = item_embed.to(device)
cf_vecs = cf_vecs.to(device)


# ============================================================
# Scoring Functions
# ============================================================
def score_candidates_with_cf(history, candidates):
    if len(history) == 0:
        return {b: 0.0 for b in candidates}

    hist_idx = [book_to_idx[b] for b in history if b in book_to_idx]
    if len(hist_idx) == 0:
        return {b: 0.0 for b in candidates}

    scores = {}
    for c in candidates:
        if c not in book_to_idx:
            scores[c] = 0.0
            continue
        ci = book_to_idx[c]
        sims = [item_sim[ci, hi] for hi in hist_idx]
        scores[c] = np.mean(sims)
    return scores


def score_with_rl(user_id, candidates):
    if user_id not in user_to_idx:
        return np.zeros(len(candidates))

    uid = user_to_idx[user_id]
    state_vec = torch.cat([user_embed[uid], torch.zeros(50, device=device)])

    item_vec_list = []
    valid_idx = []

    for i, b in enumerate(candidates):
        if b not in book_to_idx:
            continue
        bi = book_to_idx[b]
        item_vec = item_embed[bi]
        cf_vec = cf_vecs[bi]
        full_vec = torch.cat([item_vec, cf_vec])
        item_vec_list.append(full_vec)
        valid_idx.append(i)

    if len(item_vec_list) == 0:
        return np.zeros(len(candidates))

    V = torch.stack(item_vec_list)
    S = state_vec.unsqueeze(0).expand(len(V), -1)
    SA = S * V

    with torch.no_grad():
        q_vals = qnet(SA).squeeze(1)

    scores = np.zeros(len(candidates))
    q = q_vals.cpu().numpy()
    for k, idx in enumerate(valid_idx):
        scores[idx] = q[k]
    return scores


# ============================================================
# Z-score Normalization
# ============================================================
def zscore_dict(d):
    vals = np.array(list(d.values()))
    mean = vals.mean()
    std = vals.std() + 1e-9
    return {k: (v - mean) / std for k, v in d.items()}


def zscore_array(arr):
    mean = arr.mean()
    std = arr.std() + 1e-9
    return (arr - mean) / std


# ============================================================
# Evaluation
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING THREE STRATEGIES (Z-score, Seed=42)")
print("=" * 60)

# -----------------------------
# 1. Pure Item-CF
# -----------------------------
print("\n1. Pure Item-CF")
cf_hits = 0
cf_ndcg = 0
cf_prec = 0  # <-- added

for u, info in tqdm(eval_data.items(), desc="CF Eval"):
    gt = info["gt"]
    cands = [gt] + info["neg_items"]

    hist = user_hist.get(u, [])
    if len(hist) == 0:
        continue

    sc = score_candidates_with_cf(hist, cands)
    ranked = sorted(cands, key=lambda b: sc[b], reverse=True)

    if gt in ranked[:5]:
        cf_hits += 1
        idx = ranked.index(gt)
        cf_ndcg += 1.0 / np.log2(idx + 2)
        cf_prec += 1 / 5  # <-- added

cf_hr = cf_hits / len(eval_data)
cf_ndcg_score = cf_ndcg / len(eval_data)
cf_prec_score = cf_prec / len(eval_data)

print(f"   HR@5:   {cf_hr:.4f}")
print(f"   NDCG@5:{cf_ndcg_score:.4f}")
print(f"   P@5:   {cf_prec_score:.4f}")

# -----------------------------
# 2. Pure RL
# -----------------------------
print("\n2. Pure RL")
rl_hits = 0
rl_ndcg = 0
rl_prec = 0  # <-- added

for u, info in tqdm(eval_data.items(), desc="RL Eval"):
    gt = info["gt"]
    cands = [gt] + info["neg_items"]

    scores = score_with_rl(u, cands)
    ranked = [b for _, b in sorted(zip(scores, cands), reverse=True)]

    if gt in ranked[:5]:
        rl_hits += 1
        idx = ranked.index(gt)
        rl_ndcg += 1.0 / np.log2(idx + 2)
        rl_prec += 1 / 5  # <-- added

rl_hr = rl_hits / len(eval_data)
rl_ndcg_score = rl_ndcg / len(eval_data)
rl_prec_score = rl_prec / len(eval_data)

print(f"   HR@5:   {rl_hr:.4f}")
print(f"   NDCG@5:{rl_ndcg_score:.4f}")
print(f"   P@5:   {rl_prec_score:.4f}")

# -----------------------------
# 3. Hybrid (CF + RL) — Z-score
# -----------------------------
print("\n3. Hybrid (CF + RL) — Z-score")
hybrid_hits = 0
hybrid_ndcg = 0
hybrid_prec = 0

CF_WEIGHT = 0.7

for u, info in tqdm(eval_data.items(), desc="Hybrid Eval"):
    gt = info["gt"]
    cands = [gt] + info["neg_items"]

    hist = user_hist.get(u, [])
    if len(hist) == 0:
        continue

    cf_scores = score_candidates_with_cf(hist, cands)
    rl_scores = score_with_rl(u, cands)

    cf_z = zscore_dict(cf_scores)
    rl_z = zscore_array(rl_scores)

    blended = {b: CF_WEIGHT * cf_z[b] + (1 - CF_WEIGHT) * rl_z[i] for i, b in enumerate(cands)}

    ranked = sorted(cands, key=lambda b: blended[b], reverse=True)

    if gt in ranked[:5]:
        hybrid_hits += 1
        idx = ranked.index(gt)
        hybrid_ndcg += 1.0 / np.log2(idx + 2)
        hybrid_prec += 1 / 5  # <-- added

hybrid_hr = hybrid_hits / len(eval_data)
hybrid_ndcg_score = hybrid_ndcg / len(eval_data)
hybrid_prec_score = hybrid_prec / len(eval_data)

# ============================================================
# FINAL RESULTS
# ============================================================
print("\n" + "=" * 60)
print("📊 FINAL COMPARISON (Z-score, Seed=42)")
print("=" * 60)
print(f"\n{'Strategy':<20}{'HR@5':<10}{'NDCG@5':<10}{'P@5':<10}")
print("-" * 60)
print(f"{'Item-CF (pure)':<20}{cf_hr:<10.4f}{cf_ndcg_score:<10.4f}{cf_prec_score:<10.4f}")
print(f"{'RL (pure)':<20}{rl_hr:<10.4f}{rl_ndcg_score:<10.4f}{rl_prec_score:<10.4f}")
print(
    f"{'Hybrid (CF+RL)':<20}{hybrid_hr:<10.4f}{hybrid_ndcg_score:<10.4f}{hybrid_prec_score:<10.4f}"
)
print("=" * 60)

# ============================================================
# SAVE HYBRID RERANKER (BEST MODEL)
# ============================================================

print("\nSaving Hybrid Z-score Reranker...")

hybrid_model = {
    "CF_WEIGHT": CF_WEIGHT,
    "description": "Hybrid Z-score reranker = CF_z * w + RL_z * (1-w)",
}

os.makedirs(MODELS, exist_ok=True)
save_path = f"{MODELS}/hybrid_reranker.pkl"
with open(save_path, "wb") as f:
    pickle.dump(hybrid_model, f)

print(f"Hybrid reranker saved to: {save_path}")
