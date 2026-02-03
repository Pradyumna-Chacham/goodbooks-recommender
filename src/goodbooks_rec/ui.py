import pandas as pd
import streamlit as st

IMAGE_COL = "image_url"
FALLBACK_IMAGE_COL = "small_image_url"
PER_PAGE = 10


def render_book_recommendations(
    book_ids: list, artifacts, page_key: str = "rec_page", show_interaction_count: bool = False
):
    """
    Render a paginated list of book recommendations.

    Args:
        book_ids: List of book IDs to display
        artifacts: Artifacts object containing books DataFrame with metadata
        page_key: Unique key for pagination state (use different keys for multiple lists on same page)
        show_interaction_count: Whether to show "Based on X user interactions" caption
    """
    books = artifacts.books

    # Filter books to only those in book_ids, maintaining order
    # Create a mapping for efficient lookup
    book_id_to_idx = {bid: idx for idx, bid in enumerate(book_ids)}
    recs = books[books["book_id"].isin(book_ids)].copy()

    # Sort by the order in book_ids
    recs["_sort_order"] = recs["book_id"].map(book_id_to_idx)
    recs = recs.sort_values("_sort_order").drop(columns=["_sort_order"])

    total = len(recs)
    if total == 0:
        st.warning("No recommendations found.")
        return

    total_pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)

    # Initialize pagination state
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    # Navigation buttons
    nav_l, nav_m, nav_r = st.columns([1, 4, 1])

    with nav_l:
        if st.button("⬅️ Prev", key=f"{page_key}_prev", disabled=(st.session_state[page_key] <= 1)):
            st.session_state[page_key] -= 1
            st.rerun()

    with nav_r:
        if st.button(
            "Next ➡️", key=f"{page_key}_next", disabled=(st.session_state[page_key] >= total_pages)
        ):
            st.session_state[page_key] += 1
            st.rerun()

    # Clamp page number
    st.session_state[page_key] = max(1, min(st.session_state[page_key], total_pages))

    with nav_m:
        st.caption(f"Page {st.session_state[page_key]} of {total_pages}")

    # Compute slice
    start = (st.session_state[page_key] - 1) * PER_PAGE
    end = min(start + PER_PAGE, total)

    st.caption(f"Showing {start + 1}–{end} of {total} entries")

    page_recs = recs.iloc[start:end]

    # Render book cards
    cards = st.container()

    with cards:
        for _, row in page_recs.iterrows():
            img = row.get(IMAGE_COL) or row.get(FALLBACK_IMAGE_COL)

            title = str(row.get("title", "Untitled"))
            authors = str(row.get("authors", "")).strip()
            year = row.get("original_publication_year", None)
            avg_rating = row.get("average_rating", None)
            ratings_count = row.get("ratings_count", None)

            cols = st.columns([1, 4])

            with cols[0]:
                if isinstance(img, str) and img:
                    st.image(img, width=90)
                else:
                    st.write("📕")

            with cols[1]:
                st.subheader(title)

                meta_parts = []
                if authors:
                    meta_parts.append(authors)
                if year and str(year) != "nan":
                    try:
                        meta_parts.append(str(int(float(year))))
                    except Exception:
                        meta_parts.append(str(year))

                if meta_parts:
                    st.caption(" • ".join(meta_parts))

                if avg_rating is not None and str(avg_rating) != "nan":
                    st.write(f"⭐{float(avg_rating):.2f} Average Rating")
                if ratings_count is not None and str(ratings_count) != "nan":
                    try:
                        st.caption(f"{int(float(ratings_count)):,} ratings")
                    except Exception:
                        st.caption(f"{ratings_count} ratings")

                # Optional: show interaction count if provided
                if (
                    show_interaction_count
                    and "n_ratings" in recs.columns
                    and row.get("n_ratings") is not None
                ):
                    try:
                        st.caption(f"Based on {int(row['n_ratings']):,} user interactions")
                    except Exception:
                        pass

            st.divider()
