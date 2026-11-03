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
DATA_DIR  = os.path.join(BASE_DIR, "data")
MOVIE_LIST_PATH = os.path.join(MODEL_DIR, "movie_list.pkl")
SIMILARITY_PATH  = os.path.join(MODEL_DIR, "similarity.pkl")

# ── TMDB API Key — st.secrets first (Streamlit Cloud), os.getenv fallback ────
def _resolve_api_key() -> str:
    _PLACEHOLDERS = {"your_tmdb_api_key_here", "none", ""}
    try:
        raw = st.secrets.get("TMDB_API_KEY", "").strip()
        if raw.lower() not in _PLACEHOLDERS:
            return raw
    except Exception:
        pass
    raw = os.getenv("TMDB_API_KEY", "").strip()
    if raw.lower() not in _PLACEHOLDERS:
        return raw
    return ""

TMDB_API_KEY = _resolve_api_key()

# ── Fallback SVG Poster ───────────────────────────────────────────────────────
PLACEHOLDER_POSTER = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='450' viewBox='0 0 300 450'>"
    "<defs>"
    "<linearGradient id='bg' x1='0' y1='0' x2='0' y2='1'>"
    "<stop offset='0%' stop-color='%230F3460'/>"
    "<stop offset='100%' stop-color='%2316213E'/>"
    "</linearGradient>"
    "</defs>"
    "<rect width='300' height='450' fill='url(%23bg)'/>"
    "<rect x='20' y='20' width='260' height='410' rx='8' fill='none' stroke='%23E94560' stroke-width='1' stroke-opacity='0.3'/>"
    "<circle cx='150' cy='170' r='48' fill='%2316213E' stroke='%23E94560' stroke-width='1.5'/>"
    "<polygon points='140,153 173,170 140,187' fill='%23E94560'/>"
    "<text x='150' y='255' font-family='Arial,sans-serif' font-size='15' fill='%23EAEAEA' font-weight='bold' text-anchor='middle'>No Poster</text>"
    "<text x='150' y='277' font-family='Arial,sans-serif' font-size='11' fill='%23777' text-anchor='middle'>Available</text>"
    "</svg>"
)

# ── Global CSS Injection ──────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Google Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── CSS Variables ── */
        :root {
            --red:      #E94560;
            --red-dark: #c73652;
            --bg:       #0F3460;
            --bg2:      #16213E;
            --bg3:      #0a1628;
            --text:     #EAEAEA;
            --muted:    #8899aa;
            --border:   rgba(233,69,96,0.18);
            --card-bg:  rgba(22,33,62,0.85);
            --glow:     rgba(233,69,96,0.25);
        }

        /* ── Base ── */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif !important;
            background: var(--bg3) !important;
        }
        [data-testid="stAppViewContainer"] > .main {
            background: linear-gradient(160deg, #0a1628 0%, #0F3460 55%, #16213E 100%);
            min-height: 100vh;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1628 0%, #0F3460 100%) !important;
            border-right: 1px solid var(--border) !important;
        }
        [data-testid="stSidebar"] * { color: var(--text) !important; }

        /* ── Hide default Streamlit chrome ── */
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 1.5rem !important; }

        /* ── Hero title ── */
        .nf-hero {
            text-align: center;
            padding: 2.5rem 1rem 1.5rem;
        }
        .nf-hero h1 {
            font-size: clamp(2.4rem, 5vw, 3.8rem);
            font-weight: 700;
            letter-spacing: -1px;
            line-height: 1.1;
            background: linear-gradient(135deg, #fff 0%, #E94560 60%, #ff8c42 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 0.4rem;
        }
        .nf-hero p {
            font-size: 1.05rem;
            color: var(--muted);
            margin: 0 0 2rem;
            letter-spacing: 0.02em;
        }

        /* ── Metric pills ── */
        .nf-pills {
            display: flex;
            gap: 0.75rem;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 0.5rem;
        }
        .nf-pill {
            background: rgba(233,69,96,0.10);
            border: 1px solid rgba(233,69,96,0.30);
            border-radius: 999px;
            padding: 0.35rem 1.1rem;
            font-size: 0.82rem;
            font-weight: 500;
            color: var(--text);
            letter-spacing: 0.03em;
        }
        .nf-pill span { color: var(--red); font-weight: 700; }

        /* ── Divider ── */
        .nf-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--red), transparent);
            margin: 1.5rem 0;
            opacity: 0.4;
        }

        /* ── Search section label ── */
        .nf-search-label {
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.5rem;
        }

        /* ── Movie Cards Grid ── */
        .nf-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
            gap: 1.2rem;
            margin-top: 1.5rem;
        }

        /* ── Single movie card ── */
        .nf-card {
            position: relative;
            border-radius: 12px;
            overflow: hidden;
            background: var(--card-bg);
            border: 1px solid var(--border);
            transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
            cursor: default;
        }
        .nf-card:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 16px 40px rgba(0,0,0,0.5), 0 0 0 1px var(--red);
            border-color: var(--red);
        }

        /* poster image wrapper */
        .nf-poster {
            width: 100%;
            aspect-ratio: 2/3;
            overflow: hidden;
            display: block;
        }
        .nf-poster img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            transition: transform 0.35s ease;
        }
        .nf-card:hover .nf-poster img {
            transform: scale(1.06);
        }

        /* hover overlay */
        .nf-overlay {
            position: absolute;
            inset: 0;
            background: linear-gradient(to top,
                rgba(10,22,40,0.98) 0%,
                rgba(10,22,40,0.6) 45%,
                transparent 100%);
            opacity: 0;
            transition: opacity 0.25s ease;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            padding: 1rem 0.85rem 0.75rem;
        }
        .nf-card:hover .nf-overlay { opacity: 1; }

        .nf-overlay-score {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--red);
            line-height: 1;
            margin-bottom: 0.2rem;
        }
        .nf-overlay-label {
            font-size: 0.72rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        /* ── Card info area ── */
        .nf-card-body {
            padding: 0.75rem 0.85rem 0.9rem;
        }
        .nf-card-title {
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text);
            line-height: 1.3;
            margin: 0 0 0.55rem;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        /* score bar */
        .nf-bar-wrap {
            height: 4px;
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
            margin-bottom: 0.65rem;
            overflow: hidden;
        }
        .nf-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--red-dark), var(--red), #ff8c42);
            transition: width 0.6s ease;
        }
        .nf-score-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.65rem;
        }
        .nf-score-pct {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--red);
        }
        .nf-score-cap {
            font-size: 0.7rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        /* TMDB button */
        .nf-tmdb-btn {
            display: block;
            text-align: center;
            padding: 0.38rem 0;
            border-radius: 6px;
            border: 1px solid rgba(233,69,96,0.35);
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--red) !important;
            text-decoration: none !important;
            letter-spacing: 0.04em;
            transition: background 0.18s, border-color 0.18s, color 0.18s;
        }
        .nf-tmdb-btn:hover {
            background: var(--red);
            border-color: var(--red);
            color: #fff !important;
        }

        /* ── Section header ── */
        .nf-section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.01em;
            margin: 0 0 0.2rem;
        }
        .nf-section-sub {
            font-size: 0.82rem;
            color: var(--muted);
            margin: 0 0 1rem;
        }

        /* ── Sidebar stat cards ── */
        .nf-stat {
            background: rgba(233,69,96,0.07);
            border: 1px solid rgba(233,69,96,0.18);
            border-radius: 10px;
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.6rem;
        }
        .nf-stat-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--red);
            line-height: 1;
        }
        .nf-stat-label {
            font-size: 0.72rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.15rem;
        }

        /* ── Sidebar API badge ── */
        .nf-api-on  { color: #4ade80; font-weight: 600; font-size: 0.85rem; }
        .nf-api-off { color: #f59e0b; font-weight: 600; font-size: 0.85rem; }

        /* ── Streamlit widget overrides ── */
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"]    label { font-size: 0.8rem !important; color: var(--muted) !important; }

        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, var(--red-dark), var(--red)) !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            letter-spacing: 0.03em !important;
            transition: opacity 0.18s, transform 0.18s !important;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            opacity: 0.9 !important;
            transform: translateY(-1px) !important;
        }
        div[data-testid="stButton"] button[kind="secondary"] {
            background: rgba(233,69,96,0.10) !important;
            border: 1px solid rgba(233,69,96,0.35) !important;
            border-radius: 10px !important;
            color: var(--red) !important;
            font-weight: 600 !important;
        }
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background: rgba(233,69,96,0.22) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Card HTML builder ─────────────────────────────────────────────────────────
def _movie_card_html(rec: dict) -> str:
    """Return the full HTML for a single movie card."""
    title   = rec["title"].replace('"', "&quot;").replace("<", "&lt;")
    score   = rec["score"]
    bar_pct = min(score, 100.0)
    poster  = rec["poster"]
    tmdb    = f"https://www.themoviedb.org/movie/{rec['movie_id']}"

    return f"""
    <div class="nf-card">
        <div class="nf-poster">
            <img src="{poster}" alt="{title}" loading="lazy"
                 onerror="this.src='{PLACEHOLDER_POSTER}'"/>
        </div>
        <div class="nf-overlay">
            <div class="nf-overlay-score">{score}%</div>
            <div class="nf-overlay-label">match score</div>
        </div>
        <div class="nf-card-body">
            <div class="nf-card-title">{title}</div>
            <div class="nf-score-row">
                <span class="nf-score-pct">{score}%</span>
                <span class="nf-score-cap">match</span>
            </div>
            <div class="nf-bar-wrap">
                <div class="nf-bar-fill" style="width:{bar_pct}%"></div>
            </div>
            <a class="nf-tmdb-btn" href="{tmdb}" target="_blank" rel="noopener">
                View on TMDB ↗
            </a>
        </div>
    </div>
    """


# ── Load Model (Cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading recommendation engine…")
def load_data():
    """Load precomputed movie list and cosine similarity matrix.

    Falls back to auto-generation from raw CSVs if pkl files are missing.
    """
    if not os.path.exists(MOVIE_LIST_PATH) or not os.path.exists(SIMILARITY_PATH):
        m_csv = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
        c_csv = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
        if os.path.exists(m_csv) and os.path.exists(c_csv):
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
        sim = pickle.load(f)

    movies_df = movies_df.reset_index(drop=True)
    return movies_df, sim


# ── Poster Fetching (Cached 24 h) ────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=86_400)
def fetch_poster(movie_id):
    """Fetch poster URL from TMDB; return SVG fallback on any error."""
    if not TMDB_API_KEY or not movie_id:
        return PLACEHOLDER_POSTER
    try:
        url    = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        hdrs   = {"User-Agent": "NextFlick-MovieRecommender/2.0"}
        resp   = requests.get(url, params=params, headers=hdrs, timeout=(3, 5))
        if resp.status_code == 200:
            p = resp.json().get("poster_path")
            if p:
                return f"https://image.tmdb.org/t/p/w500{p}"
    except Exception:
        pass
    return PLACEHOLDER_POSTER


# ── Recommendation Engine ────────────────────────────────────────────────────
def recommend(movie_title: str, movies_df, similarity_matrix, top_n: int = 5) -> list[dict]:
    """Return top_n recommended movies. Empty list when movie not found."""
    matches = movies_df[movies_df["title"] == movie_title]
    if matches.empty:
        return []
    movie_idx = matches.index[0]
    if movie_idx >= len(similarity_matrix):
        return []
    ranked = sorted(enumerate(similarity_matrix[movie_idx]), key=lambda x: x[1], reverse=True)
    results: list[dict] = []
    for row_idx, score in ranked[1 : top_n + 1]:
        rec = movies_df.iloc[row_idx]
        results.append(
            {
                "title":    rec["title"],
                "movie_id": rec["movie_id"],
                "poster":   fetch_poster(rec["movie_id"]),
                "score":    round(float(score) * 100, 1),
            }
        )
    return results


# ── Main Application ─────────────────────────────────────────────────────────
def main() -> None:
    inject_css()
    movies, similarity = load_data()

    # ── Error state ───────────────────────────────────────────────────────────
    if movies is None or similarity is None:
        st.markdown(
            "<div class='nf-hero'><h1>🎬 NextFlick</h1>"
            "<p>Model artifacts not found.</p></div>",
            unsafe_allow_html=True,
        )
        st.error(
            "**Model files missing.**  \n"
            "Run `python generate_model.py` locally, commit the `model/` folder, then redeploy."
        )
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎬 NextFlick")
        st.markdown("<div class='nf-divider'></div>", unsafe_allow_html=True)

        st.markdown("**⚙️ Settings**")
        top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5, step=1)

        st.markdown("<div class='nf-divider'></div>", unsafe_allow_html=True)
        st.markdown("**📡 Poster API**")
        if TMDB_API_KEY:
            st.markdown(
                "<p class='nf-api-on'>● Live posters enabled</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p class='nf-api-off'>● SVG placeholders (no key)</p>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Add `TMDB_API_KEY` to Streamlit Secrets or your `.env` file.  \n"
                "[Get a free key →](https://www.themoviedb.org/settings/api)"
            )

        st.markdown("<div class='nf-divider'></div>", unsafe_allow_html=True)
        st.markdown("**📊 Catalog**")
        st.markdown(
            f"<div class='nf-stat'>"
            f"<div class='nf-stat-value'>{len(movies):,}</div>"
            f"<div class='nf-stat-label'>Movies indexed</div>"
            f"</div>"
            f"<div class='nf-stat'>"
            f"<div class='nf-stat-value'>5,000</div>"
            f"<div class='nf-stat-label'>BoW features</div>"
            f"</div>"
            f"<div class='nf-stat'>"
            f"<div class='nf-stat-value'>5</div>"
            f"<div class='nf-stat-label'>Content signals</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='nf-divider'></div>", unsafe_allow_html=True)
        st.caption("Built with [Streamlit](https://streamlit.io) · Data by [TMDB](https://www.themoviedb.org)")

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="nf-hero">
            <h1>🎬 NextFlick</h1>
            <p>Content-based movie recommendations · Bag-of-Words · Cosine Similarity</p>
            <div class="nf-pills">
                <div class="nf-pill"><span>{len(movies):,}</span> movies</div>
                <div class="nf-pill">Bag-of-Words · <span>5k</span> features</div>
                <div class="nf-pill"><span>5</span> content signals</div>
                <div class="nf-pill">Cosine Similarity</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='nf-divider'></div>", unsafe_allow_html=True)

    # ── Movie selector ────────────────────────────────────────────────────────
    movie_titles = list(movies["title"].values)
    if "selected_movie" not in st.session_state:
        st.session_state.selected_movie = movie_titles[0] if movie_titles else ""

    col_select, col_rand = st.columns([5, 1])

    with col_rand:
        st.write("")
        st.write("")
        if st.button("🎲 Surprise", use_container_width=True):
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

    if st.button("🔍 Find Similar Movies", type="primary", use_container_width=True):
        with st.spinner(f"Finding movies similar to **{selected_movie}**…"):
            results = recommend(selected_movie, movies, similarity, top_n=top_n)

        if not results:
            st.warning(f"No recommendations found for **{selected_movie}**.")
            return

        # ── Results header ────────────────────────────────────────────────────
        st.markdown("<div class='nf-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='nf-section-title'>Because you liked <em>{selected_movie}</em></div>"
            f"<div class='nf-section-sub'>Top {len(results)} content-based matches · sorted by similarity score</div>",
            unsafe_allow_html=True,
        )

        # ── Cards grid ────────────────────────────────────────────────────────
        cards_html = "".join(_movie_card_html(r) for r in results)
        st.markdown(
            f"<div class='nf-grid'>{cards_html}</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
