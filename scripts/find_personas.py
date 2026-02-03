import pandas as pd

TRAIN = "new_data/train_df.pkl"
GENRES = "data/book_genres.csv"

MIN_TOTAL = 20
MIN_SHARE = 0.30

GENRE_BUCKETS = {
    "Fantasy": ["fantasy", "ya-fantasy", "paranormal-fantasy", "fantasy-sci-fi"],
    "Sci-Fi": [
        "science-fiction",
        "scifi",
        "sci-fi",
        "scifi-fantasy",
        "sci-fi-fantasy",
        "speculative-fiction",
    ],
    "History": ["history", "historical", "historical-fiction"],
    "Romance": ["romance", "paranormal-romance", "ya-romance", "romantic"],
    "Mystery / Thriller": ["mystery", "thriller", "crime", "detective", "suspense"],
    "YA / Teen": ["young-adult", "ya", "ya-fiction", "teen", "teen-fiction"],
    "Classics": ["classics", "classic-literature", "literary-fiction", "modern-classics"],
}


def map_buckets(genres_str: str):
    s = str(genres_str).lower()
    buckets = []
    for bucket, kws in GENRE_BUCKETS.items():
        if any(kw in s for kw in kws):
            buckets.append(bucket)
    return buckets


def main():
    train = pd.read_pickle(TRAIN)
    train["user_id"] = train["user_id"].astype(int)
    train["book_id"] = train["book_id"].astype(int)

    # Use unique (user, book) so repeats don't inflate totals
    ub = train[["user_id", "book_id"]].drop_duplicates()

    genres = pd.read_csv(GENRES)
    genres["book_id"] = genres["book_id"].astype(int)

    # Map each book -> list of buckets (no explode into tags)
    b = genres[["book_id", "genres"]].copy()
    b["bucket_list"] = b["genres"].apply(map_buckets)

    # Explode buckets (few per book), not raw tags (many per book)
    b = b.explode("bucket_list").rename(columns={"bucket_list": "bucket"})
    b = b[b["bucket"].notna()]

    # Merge (user, book) with (book, bucket)
    df = ub.merge(b[["book_id", "bucket"]], on="book_id", how="inner").drop_duplicates()

    totals = ub.groupby("user_id").size().rename("total").reset_index()
    counts = df.groupby(["user_id", "bucket"]).size().rename("n").reset_index()
    counts = counts.merge(totals, on="user_id", how="left")
    counts["share"] = counts["n"] / counts["total"]

    print("\n=== SELECTED PERSONAS (ONE PER BUCKET) ===\n")
    selected = {}

    for bucket in GENRE_BUCKETS.keys():
        sub = counts[
            (counts["bucket"] == bucket)
            & (counts["total"] >= MIN_TOTAL)
            & (counts["share"] >= MIN_SHARE)
        ].copy()

        if sub.empty:
            # fallback: pick best available for the bucket (still useful)
            sub = counts[(counts["bucket"] == bucket) & (counts["total"] >= MIN_TOTAL)].copy()
            if sub.empty:
                print(f"❌ {bucket}: no candidate found")
                continue

        best = sub.sort_values(["share", "total"], ascending=[False, False]).iloc[0]
        uid = int(best["user_id"])
        selected[bucket] = uid

        print(f"✅ {bucket}: user_id={uid} (share={best['share']:.2f}, total={int(best['total'])})")

    print("\n--- COPY INTO apps/personas.py ---\n")
    print("PERSONAS = {")
    print('    "🎲 Random active user": None,')
    for bucket, uid in selected.items():
        print(f'    "{bucket}": {uid},')
    print("}")


if __name__ == "__main__":
    main()
