from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


from .config import DATA_DIR,MODELS_DIR

@dataclass(frozen=True)
class Artifacts:
    books:pd.DataFrame
    ratings: pd.DataFrame 
    model_popularity: object | None=None


def load_artifacts(data_dir: Path=DATA_DIR,models_dir=MODELS_DIR) -> Artifacts:
    books_path=data_dir/"books.csv"
    ratings_path=data_dir/"ratings.csv"

    if not books_path.exists() or not ratings_path.exists():
        raise FileNotFoundError("Missing data files.Expected books.csv and ratings.csv under ./data.If you didn't commit the dataset, download it and place it in data/")
    
    books=pd.read_csv(books_path)
    ratings=pd.read_csv(ratings_path)

    pop_path=models_dir/"popularity.pkl"
    model_popularity=None

    if pop_path.exists():
        with open(pop_path,"rb") as f:
            model_popularity=pickle.load(f)

    return Artifacts(books=books,ratings=ratings,model_popularity=model_popularity)

