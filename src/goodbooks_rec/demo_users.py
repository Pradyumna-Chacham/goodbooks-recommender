import random
from typing import Dict, List


def get_valid_user_ids(A: Dict) -> List[int]:
    # user_hist keys are actual user_ids present in training data
    return sorted(A["user_hist"].keys())


def sample_random_user(A: Dict) -> int:
    ids = get_valid_user_ids(A)
    return random.choice(ids) if ids else 1
