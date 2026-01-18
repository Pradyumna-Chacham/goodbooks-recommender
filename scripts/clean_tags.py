"""
clean_tags.py
-------------------------------------
Cleans Goodbooks tags into usable genre/theme features.

Inputs (expected in ./data):
    - tags.csv
    - book_tags.csv
    - books.csv

Outputs (in ./new_data):
    - book_genres.csv
        columns: [book_id, goodreads_book_id, genres]
    - book_genres.pkl
        dict: book_id -> list of genres
    - book_metadata.csv
        columns: [book_id, goodreads_book_id, title, authors, genres]
"""

import os
import re
import pickle
from collections import Counter

import pandas as pd

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DATA_DIR = "data"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

FREQ_MIN = 30  # minimum number of distinct books per tag

# Reader-behavior / format / shelf tags to always drop
BLACKLIST = {
    "to-read", "favorites", "favourites", "owned", "kindle",
    "audiobook", "audio", "ebook", "e-book", "paperback", "hardcover",
    "currently-reading", "did-not-finish", "dnf", "tbr",
    "library", "my-library", "book-club", "bookclub", "wishlist",
}

# Genre / theme keywords we consider "semantic"
GENRE_KEYWORDS = [
    # fantasy & related
    "fantasy", "magic", "dragon", "wizard", "witch",
    "epic-fantasy", "urban-fantasy", "dark-fantasy", "high-fantasy",
    # sci-fi & related
    "sci-fi", "scifi", "science-fiction", "space", "space-opera",
    "dystopian", "post-apocalyptic", "aliens", "cyberpunk",
    # crime / mystery / thriller
    "mystery", "crime", "detective", "thriller", "suspense", "noir",
    # romance & relationships
    "romance", "love-story", "contemporary-romance", "historical-romance",
    # horror & dark
    "horror", "ghost", "gothic", "creepy",
    # historical / classics / literature
    "historical", "historical-fiction", "classics", "literature",
    "literary-fiction",
    # general fiction / contemporary
    "fiction", "contemporary", "contemporary-fiction",
    # YA / children
    "young-adult", "ya", "teen", "children", "middle-grade",
    # non-fiction & topics
    "non-fiction", "nonfiction", "biography", "memoir",
    "philosophy", "psychology", "economics", "politics", "history",
    # formats that still imply content style
    "graphic-novel", "comics", "manga",
    # myth / legend
    "mythology", "folklore", "fairy-tales", "folktales",
    # war / military
    "war", "military",
    # poetry
    "poetry",
]

# Normalize GENRE_KEYWORDS
def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\- ]", "", s)
    s = s.strip()
    return s

GENRE_KEYWORDS = [_norm(k) for k in GENRE_KEYWORDS]


# Subgenre mapping to coarser genres (optional consolidation)
SUBGENRE_MAP = {
    "epic-fantasy": "fantasy",
    "urban-fantasy": "fantasy",
    "dark-fantasy": "fantasy",
    "high-fantasy": "fantasy",
    "space-opera": "science-fiction",
    "dystopian": "science-fiction",
    "post-apocalyptic": "science-fiction",
    "crime-fiction": "mystery",
    "psych-thriller": "thriller",
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def normalize_tag(t: str) -> str:
    """Lowercase, remove punctuation except hyphen/space, trim."""
    t = str(t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9\- ]", "", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip()
    return t


def is_blacklisted(t: str) -> bool:
    return t in BLACKLIST


def has_genre_keyword(t: str) -> bool:
    """Does this tag contain any of our semantic genre keywords?"""
    for kw in GENRE_KEYWORDS:
        if kw and kw in t:
            return True
    return False


def map_subgenre(tag: str) -> str:
    """Map some niche subgenres to main genres if present."""
    if tag in SUBGENRE_MAP:
        return SUBGENRE_MAP[tag]
    # also try substring-based mapping
    for sub, parent in SUBGENRE_MAP.items():
        if sub in tag:
            return parent
    return tag


# -------------------------------------------------------------------
# Main cleaning logic
# -------------------------------------------------------------------
def main():
    print("Loading raw data...")

    tags = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))
    book_tags = pd.read_csv(os.path.join(DATA_DIR, "book_tags.csv"))
    books = pd.read_csv(os.path.join(DATA_DIR, "books.csv"))

    # Merge tag names into book_tags
    merged = book_tags.merge(tags, on="tag_id", how="left")

    print(f"Raw book_tags rows: {len(merged):,}")

    # Normalize tags
    merged["clean_tag"] = merged["tag_name"].apply(normalize_tag)

    # Count how many DISTINCT books each tag appears on
    tag_book_counts = merged.groupby("clean_tag")["goodreads_book_id"].nunique()
    print(f"Unique normalized tags: {len(tag_book_counts):,}")

    # Frequency filter
    frequent_tags = set(
        tag for tag, cnt in tag_book_counts.items() if cnt >= FREQ_MIN
    )
    print(f"Tags with >= {FREQ_MIN} books: {len(frequent_tags):,}")

    # Apply blacklist + genre semantic filter
    valid_tags = set()
    for t in frequent_tags:
        if is_blacklisted(t):
            continue
        if has_genre_keyword(t):
            valid_tags.add(t)

    print(f"Tags passing frequency + semantic filter: {len(valid_tags):,}")

    # Keep only rows with valid_tags
    filtered = merged[merged["clean_tag"].isin(valid_tags)].copy()
    print(f"Filtered book_tags rows: {len(filtered):,}")

    # Map subgenres to parent genres
    filtered["genre_tag"] = filtered["clean_tag"].apply(map_subgenre)

    # Build per-goodreads_book_id genre lists
    book_genre_series = (
        filtered.groupby("goodreads_book_id")["genre_tag"]
        .apply(lambda tags: sorted(set(tags)))
    )

    print(f"Books with at least one genre tag: {len(book_genre_series):,}")

    # ----------------------------------------------------------------
    # Join with books to get internal book_id, title, authors
    # ----------------------------------------------------------------
    books_small = books[["book_id", "goodreads_book_id", "title", "authors"]].copy()
    book_genres_df = book_genre_series.reset_index()
    book_genres_df["genres"] = book_genres_df["genre_tag"].apply(
        lambda lst: ",".join(lst)
    )
    book_genres_df = book_genres_df.drop(columns=["genre_tag"])

    # Merge to attach book_id/title/authors
    metadata = books_small.merge(
        book_genres_df, on="goodreads_book_id", how="inner"
    )

    print(f"Books in metadata with genres: {len(metadata):,}")

    # ----------------------------------------------------------------
    # Save book_genres (minimal, per internal book_id)
    # ----------------------------------------------------------------
    book_genres_min = metadata[["book_id", "goodreads_book_id", "genres"]].copy()
    book_genres_min.to_csv(os.path.join(OUT_DIR, "book_genres.csv"), index=False)

    # Also pickle a dict: book_id -> [genre list]
    book_id_to_genres = {
        int(row.book_id): row.genres.split(",") if row.genres else []
        for _, row in book_genres_min.iterrows()
    }
    with open(os.path.join(OUT_DIR, "book_genres.pkl"), "wb") as f:
        pickle.dump(book_id_to_genres, f)

    # ----------------------------------------------------------------
    # Save full book_metadata.csv
    # ----------------------------------------------------------------
    metadata.to_csv(os.path.join(OUT_DIR, "book_metadata.csv"), index=False)

    print("\nDone!")
    print(f"Saved: {OUT_DIR}/book_genres.csv")
    print(f"Saved: {OUT_DIR}/book_genres.pkl")
    print(f"Saved: {OUT_DIR}/book_metadata.csv")


if __name__ == "__main__":
    main()
