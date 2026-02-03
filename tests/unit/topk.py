# tests/unit/test_topk_arrays.py
import numpy as np


def test_topk_shapes_and_types(mini_item_topk, mini_user_topk):
    item_idx, item_sim = mini_item_topk
    user_idx, user_sim = mini_user_topk

    assert item_idx.ndim == 2 and item_sim.ndim == 2
    assert user_idx.ndim == 2 and user_sim.ndim == 2

    assert item_idx.shape == item_sim.shape
    assert user_idx.shape == user_sim.shape

    assert np.issubdtype(item_idx.dtype, np.integer)
    assert np.issubdtype(user_idx.dtype, np.integer)
    assert item_sim.dtype.kind == "f"
    assert user_sim.dtype.kind == "f"
