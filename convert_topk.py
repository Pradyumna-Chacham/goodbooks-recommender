import os
import pickle

import numpy as np
from tqdm import tqdm


def load_pkl_array(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        arr = pickle.load(f)
    if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{path} is not a square numpy.ndarray. Got type={type(arr)}, shape={getattr(arr,'shape',None)}"
        )
    return arr


def dense_to_topk(
    dense: np.ndarray,
    k: int,
    out_path: str,
    exclude_self: bool = True,
    sim_dtype=np.float16,
    idx_dtype=np.int32,
    compress: bool = True,
) -> None:
    """
    Converts dense NxN similarity matrix into Top-K neighbor lists per row.
    Saves:
      - idx: [N, K] indices of neighbors
      - sim: [N, K] similarity values

    Notes:
      - Uses argpartition (fast) then sorts the Top-K.
      - If exclude_self=True, diagonal is not allowed to appear in Top-K.
    """
    N = dense.shape[0]

    topk_idx = np.empty((N, k), dtype=idx_dtype)
    topk_sim = np.empty((N, k), dtype=sim_dtype)

    for i in tqdm(range(N), desc=f"TopK k={k}", total=N):
        row = dense[i]

        # Make a local view; avoid modifying the original matrix on disk
        # Exclude diagonal by setting it to -inf for selection
        if exclude_self:
            # If row is float dtype, we can handle -inf safely
            row_i = row.copy()
            row_i[i] = -np.inf
        else:
            row_i = row

        # Get indices of k largest values
        # argpartition returns unsorted top-k; then we sort
        idx = np.argpartition(row_i, -k)[-k:]
        # Sort those indices by similarity descending
        idx = idx[np.argsort(row_i[idx])[::-1]]

        sims = row_i[idx].astype(sim_dtype, copy=False)

        topk_idx[i] = idx.astype(idx_dtype, copy=False)
        topk_sim[i] = sims

    if compress:
        np.savez_compressed(out_path, idx=topk_idx, sim=topk_sim)
    else:
        np.savez(out_path, idx=topk_idx, sim=topk_sim)

    print(f"Saved: {out_path}")
    print(f"  idx: {topk_idx.shape} {topk_idx.dtype}")
    print(f"  sim: {topk_sim.shape} {topk_sim.dtype}")


def main():
    k = 100  # start here; 50-200 is common. K=100 is a good tradeoff.

    os.makedirs("models", exist_ok=True)

    # Item CF
    item_path = "models/item_cf.pkl"
    item_dense = load_pkl_array(item_path)
    print("Item dense:", item_dense.shape, item_dense.dtype)
    dense_to_topk(
        item_dense,
        k=k,
        out_path=f"models/item_topk_k{k}.npz",
        exclude_self=True,
        sim_dtype=np.float16,
    )

    # User CF
    user_path = "models/user_cf.pkl"
    user_dense = load_pkl_array(user_path)
    print("User dense:", user_dense.shape, user_dense.dtype)
    dense_to_topk(
        user_dense,
        k=k,
        out_path=f"models/user_topk_k{k}.npz",
        exclude_self=True,
        sim_dtype=np.float16,
    )


if __name__ == "__main__":
    main()
