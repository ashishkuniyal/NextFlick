import os
import pickle
import random
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Load environment variables (.env for local dev) ───────────────────────────
load_dotenv()

st.set_page_config(
    page_title="NextFlick — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants & Configuration ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")
MOVIE_LIST_PATH = os.path.join(MODEL_DIR, "movie_list.pkl")
SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity.pkl")

# ── TMDB API Key — st.secrets first (Streamlit Cloud), os.getenv fallback ────
def _resolve_api_key() -> str:
    """Return a clean TMDB API key, or '' if not configured."""
    _PLACEHOLDERS = {"your_tmdb_api_key_here", "none", ""}

    # 1. Streamlit Secrets (Streamlit Community Cloud)
    try:
        raw = st.secrets.get("TMDB_API_KEY", "").strip()
        if raw.lower() not in _PLACEHOLDERS:
            return raw
    except Exception:
        pass

    # 2. Environment variable / .env file (local dev)
    raw = os.getenv("TMDB_API_KEY", "").strip()
    if raw.lower() not in _PLACEHOLDERS:
        return raw

    return ""


TMDB_API_KEY = _resolve_api_key()

# ── Fallback SVG Poster (works 100% offline, no external CDN dependency) ─────
PLACEHOLDER_POSTER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='450' viewBox='0 0 300 450'>"
    "<rect width='300' height='450' fill='%230F3460'/>"
    "<circle cx='150' cy='175' r='50' fill='%2316213E'/>"
    "<polygon points='140,158 170,175 140,192' fill='%23E94560'/>"
    "<text x='150' y='260' font-family='Arial,sans-serif' font-size='16' fill='%23EAEAEA'"
    " font-weight='bold' text-anchor='middle'>No Poster</text>"
    "<text x='150' y='282' font-family='Arial,sans-serif' font-size='12' fill='%239a9a9a'"
    " text-anchor='middle'>Available</text>"
    "</svg>"
)


# ── Load Model & Data (Cached — rebuilt automatically if needed) ───────────────
@st.cache_resource(show_spinner="⏳ Loading recommendation models…")
def load_data():
    """Load precomputed movie list and cosine similarity matrix.

    Falls back to auto-generation from raw CSVs if the pkl files are missing
    (useful for first-run after a fresh clone with the data/ CSVs present).
    """
    if not os.path.exists(MOVIE_LIST_PATH) or not os.path.exists(SIMILARITY_PATH):
        movies_csv = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
        credits_csv = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
        if os.path.exists(movies_csv) and os.path.exists(credits_csv):
            try:
                from generate_model import build_model
                build_model(data_dir=DATA_DIR, model_dir=MODEL_DIR)
            except Exception:
                pass

    if not os.path.exists(MOVIE_LIST_PATH) or not os.path.exists(SIMILARITY_PATH):
        return None, None

    with open(MOVIE_LIST_PATH, "rb") as f:
        movies_df = pickle.load(f)
    with open(SIMILARITY_PATH, "rb") as f:
        similarity_matrix = pickle.load(f)

    # Guarantee index is strictly 0..N-1 to align with matrix rows
    movies_df = movies_df.reset_index(drop=True)
    return movies_df, similarity_matrix


# ── Poster Fetching (Cached 24 h) ────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=86_400)
def fetch_poster(movie_id):
    """Fetch a movie poster URL from TMDB API; return SVG fallback on any error."""
    if not TMDB_API_KEY or not movie_id:
        return PLACEHOLDER_POSTER

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        headers = {"User-Agent": "NextFlick-MovieRecommender/2.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=(3, 5))
        if resp.status_code == 200:
            poster_path = resp.json().get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass

    return PLACEHOLDER_POSTER


# ── Recommendation Engine ────────────────────────────────────────────────────
def recommend(movie_title: str, movies_df, similarity_matrix, top_n: int = 5) -> list[dict]:
    """Return *top_n* recommended movies with title, movie_id, poster URL, and score.

    Returns an empty list when the query movie is not found in the catalog.
    """
    matches = movies_df[movies_df["title"] == movie_title]
    if matches.empty:
        return []

    movie_idx = matches.index[0]
    if movie_idx >= len(similarity_matrix):
        return []

    sim_scores = similarity_matrix[movie_idx]
    ranked = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)

    recommendations: list[dict] = []
    for row_idx, score in ranked[1 : top_n + 1]:
        rec = movies_df.iloc[row_idx]
        recommendations.append(
            {
                "title": rec["title"],
                "movie_id": rec["movie_id"],
                "poster": fetch_poster(rec["movie_id"]),
                "score": round(float(score) * 100, 1),
            }
        )
    return recommendations


# ── Main Application ─────────────────────────────────────────────────────────
def main() -> None:
    movies, similarity = load_data()

    # ── Error state — model files missing ────────────────────────────────────
    if movies is None or similarity is None:
        st.title("🎬 NextFlick — Movie Recommender")
        st.error(
            "⚠️ **Model files not found.**\n\n"
            "The artifacts `model/movie_list.pkl` and `model/similarity.pkl` are missing.\n\n"
            "Generate them locally:\n"
            "```bash\npython generate_model.py\n```\n"
            "Then commit them and push, or restart the app."
        )
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        top_n = st.slider("Recommendations", min_value=3, max_value=10, value=5, step=1)

        st.markdown("---")
        st.subheader("🔑 TMDB API")
        if TMDB_API_KEY:
            st.success("✅ API key active — posters enabled")
        else:
            st.warning("⚠️ No API key — SVG placeholders shown")
            st.caption(
                "Add `TMDB_API_KEY` to **Streamlit Secrets** (cloud) "
                "or your local `.env` file.\n\n"
                "[Get a free key →](https://www.themoviedb.org/settings/api)"
            )

        st.markdown("---")
        st.subheader("📊 Dataset")
        st.write(f"**{len(movies):,}** movies in catalog")
        st.write("Features: genres · keywords · cast · director · overview")
        st.write("Algorithm: Bag-of-Words + Cosine Similarity")

        st.markdown("---")
        st.caption(
            "Built with [Streamlit](https://streamlit.io) · "
            "Data by [TMDB](https://www.themoviedb.org)"
        )

    # ── Main Header ───────────────────────────────────────────────────────────
    st.title("🎬 NextFlick")
    st.caption("Content-based movie recommendations · Bag-of-Words · Cosine Similarity")
    st.markdown("---")

    movie_titles = list(movies["title"].values)

    if "selected_movie" not in st.session_state:
        st.session_state.selected_movie = movie_titles[0] if movie_titles else ""

    col_select, col_rand = st.columns([4, 1])

    with col_rand:
        st.write("")
        st.write("")
        if st.button("🎲 Surprise me", use_container_width=True):
            st.session_state.selected_movie = random.choice(movie_titles)
            st.rerun()

    with col_select:
        try:
            current_index = movie_titles.index(st.session_state.selected_movie)
        except ValueError:
            current_index = 0

        selected_movie = st.selectbox(
            "Search or pick a movie you like:",
            movie_titles,
            index=current_index,
            key="movie_dropdown",
        )
        st.session_state.selected_movie = selected_movie

    # ── Recommend button ──────────────────────────────────────────────────────
    if st.button("🔍 Find Similar Movies", type="primary", use_container_width=True):
        with st.spinner(f"Finding movies similar to **{selected_movie}**…"):
            results = recommend(selected_movie, movies, similarity, top_n=top_n)

        if not results:
            st.warning(f"No recommendations found for **{selected_movie}**.")
            return

        st.markdown(f"### 🎯 Because you liked **{selected_movie}**")
        cols = st.columns(len(results))
        for col, rec in zip(cols, results):
            with col:
                st.image(rec["poster"], use_container_width=True)
                st.markdown(f"**{rec['title']}**")
                st.caption(f"Match: `{rec['score']}%`")
                tmdb_url = f"https://www.themoviedb.org/movie/{rec['movie_id']}"
                st.markdown(f"[View on TMDB ↗]({tmdb_url})")


if __name__ == "__main__":
    main()
