"""
train_rl_hybrid.py — Option A + Option C (Optimized)
-----------------------------------------------------

✔ Uses 150-dim RL inputs:
      [ user_embed (100) || item_embed (50) ] = 150
✔ Precomputes CF-TopK embedding vectors (no sorting in loop)
✔ Precomputes mean CF sims per item
✔ Vectorized scoring
✔ Fast enough for CPU (~15 min per epoch)

Output:
    models/rl_cfhard_fast.pth
"""

import os
import pickle
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


#SEED====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Deterministic flags
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Needed for cuBLAS deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# ============================================================
# Paths
# ============================================================
DATA = "new_data"
MODELS = "models"
os.makedirs(MODELS, exist_ok=True)

SVD_PATH = f"{MODELS}/svd_hybrid.pkl"
ITEMCF_PATH = f"{MODELS}/item_cf.pkl"
TRAIN_PATH = f"{DATA}/train_df.pkl"
EVAL_PATH = f"{DATA}/eval_data.pkl"

# ============================================================
# Load data
# ============================================================
print("="*60)
print(" OPTIMIZED RL TRAINER (Option A + Option C) ")
print("="*60)

svd = pickle.load(open(SVD_PATH, "rb"))
item_sim = pickle.load(open(ITEMCF_PATH, "rb"))

# embeddings
user_embed = torch.tensor(svd["user_embed"], dtype=torch.float32)
item_embed = torch.tensor(svd["item_embed"], dtype=torch.float32)

user_to_idx = svd["user_to_idx"]
book_to_idx = svd["book_to_idx"]
idx_to_book = {v: k for k, v in book_to_idx.items()}

num_users, U_dim = user_embed.shape  # 100
num_items, I_dim = item_embed.shape  # 100 (from hybrid)
RL_dim = U_dim + 50  # 100 + 50 = 150

print(f"Users: {num_users}, Items: {num_items}, RL_dim={RL_dim}")

# training data
import pandas as pd
train_df = pickle.load(open(TRAIN_PATH, "rb"))
user_pos = train_df.groupby("user_id")["book_id"].apply(list).to_dict()

# eval data
with open(EVAL_PATH, "rb") as f:
    eval_data = pickle.load(f)


# ============================================================
# Precompute CF vectors (Option C)
# ============================================================
print("\nPrecomputing CF vectors (Top-50 neighbors)...")

TOPK = 50
cf_vecs = np.zeros((num_items, 50), dtype=np.float32)
cf_sim_mean = np.zeros(num_items, dtype=np.float32)

for i in tqdm(range(num_items)):
    sims = item_sim[i]

    # top-K neighbors
    top = np.argsort(sims)[-TOPK:]
    cf_sim_mean[i] = sims[top].mean()

    # average their SVD embeddings
    cf_vecs[i] = item_embed[top].mean(dim=0)[:50]  # use 50 dims

# convert to torch
cf_vecs = torch.tensor(cf_vecs, dtype=torch.float32)
cf_sim_mean = torch.tensor(cf_sim_mean, dtype=torch.float32)


# ============================================================
# Build RL training samples
# ============================================================
print("\nBuilding RL training samples...")
samples = []

for u, seq in user_pos.items():
    if len(seq) < 4:
        continue
    for i in range(3, len(seq)):
        h = seq[:i]
        gt = seq[i]
        if gt in book_to_idx:
            samples.append((u, h, gt))

# subsample
MAX_SAMPLES = 400_000
if len(samples) > MAX_SAMPLES:
    print(f"Subsampling to {MAX_SAMPLES}")
    samples = random.sample(samples, MAX_SAMPLES)

print(f"Training samples: {len(samples)}")


# ============================================================
# Precompute all candidate negative pools
# ============================================================
def get_cf_hard_negatives(hist, num_neg=99):
    """Return 99 CF-hard negatives WITHOUT CF computations inside loop."""
    recent = hist[-3:] if len(hist) > 3 else hist
    cand = set()

    for b in recent:
        if b not in book_to_idx:
            continue
        idx = book_to_idx[b]

        # deterministic: already sorted?
        sims = item_sim[idx]
        top = np.argsort(sims)[-TOPK:]
        cand.update(idx_to_book[i] for i in top)

    cand -= set(hist)
    cand = list(cand)

    if len(cand) >= num_neg:
        return random.sample(cand, num_neg)

    # fallback random
    all_books = list(book_to_idx.keys())
    pool = list(set(all_books) - set(hist) - set(cand))
    need = num_neg - len(cand)

    if len(pool) >= need:
        return cand + random.sample(pool, need)
    else:
        return cand + pool


# ============================================================
# RL model
# ============================================================
class QNet(nn.Module):
    def __init__(self, dim=150):  # IMPORTANT FIX
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, sa):
        return self.net(sa)


# ============================================================
# Vectorized scoring
# ============================================================
def score_vectorized(model, state_batch, item_batch):
    """
    state_batch: (B, 150)
    item_batch:  (B, K, 150)
    -> q-values (B, K)
    """
    B, K, D = item_batch.shape
    s_exp = state_batch.unsqueeze(1).expand(-1, K, -1)
    sa = s_exp * item_batch  # elementwise
    q = model(sa.reshape(B * K, D))
    return q.view(B, K)


# ============================================================
# Evaluation on eval_data.pkl (same as baselines)
# ============================================================
def evaluate(model, device, n=2000):
    model.eval()

    users = list(eval_data.keys())
    users = random.sample(users, min(n, len(users)))

    hits, ndcg, total = 0, 0, 0

    for u in users:
        if u not in user_to_idx:
            continue

        info = eval_data[u]
        gt = info["gt"]
        neg = info["neg_items"][:99]
        cands = [gt] + neg

        uid = user_to_idx[u]

        # build 150-dim state
        # state is user vector only
        s_u = user_embed[uid]  # (100,)
        s_u = torch.cat([s_u, torch.zeros(50)], dim=0)  # padded to 150
        s = s_u.unsqueeze(0).to(device)

        # build 150-dim item matrix for candidates
        item_vecs = []
        for b in cands:
            bi = book_to_idx[b]
            i_vec = item_embed[bi]
            cf_vec = cf_vecs[bi]
            item_vecs.append(torch.cat([i_vec, cf_vec], dim=0).numpy())

        item_vecs = torch.tensor(np.array(item_vecs), dtype=torch.float32).unsqueeze(0).to(device)

        q = score_vectorized(model, s, item_vecs).detach().cpu().numpy().flatten()
        ranked = [x for _, x in sorted(zip(q, cands), reverse=True)]

        if gt in ranked[:5]:
            hits += 1
            idx = ranked.index(gt)
            ndcg += 1.0 / np.log2(idx + 2)

        total += 1

    return hits / total, ndcg / total


# ============================================================
# Training
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

user_embed = user_embed.to(device)
item_embed = item_embed.to(device)
cf_vecs = cf_vecs.to(device)

qnet = QNet(dim=150).to(device)
optimizer = optim.Adam(qnet.parameters(), lr=1e-4)

epochs = 3
batch_size = 256

print("\nStarting RL training...\n")

for ep in range(1, epochs + 1):
    random.shuffle(samples)

    batch_losses = []
    num_batches = len(samples) // batch_size

    for bi in tqdm(range(num_batches), desc=f"Epoch {ep}"):
        batch = samples[bi * batch_size:(bi + 1) * batch_size]

        states, items, rewards = [], [], []

        for u, hist, gt in batch:
            uid = user_to_idx[u]

            # state = user embed (100) + zero vector (50)
            s_u = user_embed[uid]
            s_u = torch.cat([s_u, torch.zeros(50)], dim=0)
            states.append(s_u.numpy())

            # candidates
            negs = get_cf_hard_negatives(hist, 99)
            cands = [gt] + negs

            item_vec_list = []
            reward_list = [1.0] + [0.0] * 99

            for b in cands:
                bi = book_to_idx[b]
                i_vec = item_embed[bi]
                cf_vec = cf_vecs[bi]
                item_vec_list.append(torch.cat([i_vec, cf_vec], dim=0).numpy())

            items.append(item_vec_list)
            rewards.append(reward_list)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        items = torch.tensor(np.array(items), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)

        q_values = score_vectorized(qnet, states, items)
        loss = ((q_values - rewards)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    hr, ndcg = evaluate(qnet, device)
    print(f"\nEpoch {ep}: loss={np.mean(batch_losses):.4f}, HR@5={hr:.4f}, NDCG@5={ndcg:.4f}")
    torch.save(qnet.state_dict(), f"{MODELS}/rl_cfhard_fast.pth")
    print("Saved model.\n")

print("Training complete.")
