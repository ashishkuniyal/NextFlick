"""
Smoke tests for the Movie Recommender System.
These tests run entirely offline — no network, no Streamlit server needed.

Run with:
    pytest tests/ -v
or the helper script:
    python -m pytest tests/ -v
"""

import os
import sys
import pickle
import types
import importlib
import pytest

# ── Resolve repo root so imports work regardless of CWD ─────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "model")
MOVIE_LIST_PATH = os.path.join(MODEL_DIR, "movie_list.pkl")
SIMILARITY_PATH = os.path.join(MODEL_DIR, "similarity.pkl")


# ── Helpers ──────────────────────────────────────────────────────────────────
def model_files_exist():
    return os.path.exists(MOVIE_LIST_PATH) and os.path.exists(SIMILARITY_PATH)


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def model_data():
    """Load model artifacts once for all tests in this session."""
    if not model_files_exist():
        pytest.skip("Model files not found — run `python generate_model.py` first.")
    with open(MOVIE_LIST_PATH, "rb") as f:
        movies = pickle.load(f)
    with open(SIMILARITY_PATH, "rb") as f:
        similarity = pickle.load(f)
    movies = movies.reset_index(drop=True)
    return movies, similarity


# ── Phase 1: Artifact Integrity ───────────────────────────────────────────────
class TestArtifactIntegrity:
    def test_movie_list_loads(self, model_data):
        movies, _ = model_data
        assert movies is not None, "movie_list.pkl failed to load"

    def test_similarity_loads(self, model_data):
        _, similarity = model_data
        assert similarity is not None, "similarity.pkl failed to load"

    def test_movie_list_has_required_columns(self, model_data):
        movies, _ = model_data
        required = {"movie_id", "title", "tags"}
        missing = required - set(movies.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_similarity_is_square(self, model_data):
        movies, similarity = model_data
        n = len(movies)
        assert similarity.shape == (n, n), (
            f"Shape mismatch: similarity={similarity.shape}, movies={n}"
        )

    def test_index_aligned_with_matrix(self, model_data):
        movies, similarity = model_data
        assert movies.index[0] == 0, "DataFrame index must start at 0"
        assert movies.index[-1] == len(movies) - 1, "DataFrame index must be contiguous 0..N-1"

    def test_reasonable_catalog_size(self, model_data):
        movies, _ = model_data
        assert len(movies) >= 4000, f"Expected ≥4000 movies, got {len(movies)}"

    def test_no_null_titles(self, model_data):
        movies, _ = model_data
        nulls = movies["title"].isna().sum()
        assert nulls == 0, f"Found {nulls} null titles"


# ── Phase 2: Recommendation Engine ───────────────────────────────────────────
class TestRecommendationEngine:
    """Test recommend() directly — stubs out Streamlit and network calls."""

    @pytest.fixture(autouse=True)
    def _stub_streamlit(self, monkeypatch):
        """Prevent Streamlit from being imported or called during tests."""
        fake_st = types.ModuleType("streamlit")
        fake_st.cache_data = lambda **kw: (lambda f: f)
        fake_st.cache_resource = lambda **kw: (lambda f: f)
        fake_st.set_page_config = lambda **kw: None
        fake_st.secrets = {}
        monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    @pytest.fixture(autouse=True)
    def _stub_dotenv(self, monkeypatch):
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    @pytest.fixture()
    def recommend_fn(self):
        """Import recommend() from app.py with all side-effects stubbed."""
        # Remove cached module if already imported
        sys.modules.pop("app", None)
        # Temporarily add repo root to sys.path
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        mod = importlib.import_module("app")
        return mod.recommend

    def test_recommend_returns_five(self, model_data, recommend_fn, monkeypatch):
        movies, similarity = model_data
        # Stub fetch_poster to avoid network calls
        import app as app_mod
        monkeypatch.setattr(app_mod, "fetch_poster", lambda mid: "stub_poster_url")
        results = recommend_fn("Avatar", movies, similarity, top_n=5)
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    def test_recommend_structure(self, model_data, recommend_fn, monkeypatch):
        movies, similarity = model_data
        import app as app_mod
        monkeypatch.setattr(app_mod, "fetch_poster", lambda mid: "stub_poster_url")
        results = recommend_fn("Avatar", movies, similarity, top_n=3)
        for r in results:
            assert "title" in r
            assert "movie_id" in r
            assert "poster" in r
            assert "score" in r
            assert 0.0 <= r["score"] <= 100.0

    def test_recommend_excludes_query_movie(self, model_data, recommend_fn, monkeypatch):
        movies, similarity = model_data
        import app as app_mod
        monkeypatch.setattr(app_mod, "fetch_poster", lambda mid: "stub_poster_url")
        results = recommend_fn("Avatar", movies, similarity, top_n=5)
        titles = [r["title"] for r in results]
        assert "Avatar" not in titles, "Query movie must not appear in recommendations"

    def test_recommend_unknown_movie(self, model_data, recommend_fn, monkeypatch):
        movies, similarity = model_data
        import app as app_mod
        monkeypatch.setattr(app_mod, "fetch_poster", lambda mid: "stub_poster_url")
        results = recommend_fn("__NOT_A_REAL_MOVIE__", movies, similarity)
        assert results == [], "Unknown movie should return empty list"

    def test_recommend_scores_descending(self, model_data, recommend_fn, monkeypatch):
        movies, similarity = model_data
        import app as app_mod
        monkeypatch.setattr(app_mod, "fetch_poster", lambda mid: "stub_poster_url")
        results = recommend_fn("The Dark Knight", movies, similarity, top_n=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Scores must be in descending order"


# ── Phase 3: Key Resolution ───────────────────────────────────────────────────
class TestApiKeyResolution:
    """Verify st.secrets → os.getenv fallback chain without network calls."""

    def _load_app_module(self, monkeypatch, secrets=None, env_val=None):
        sys.modules.pop("app", None)

        fake_st = types.ModuleType("streamlit")
        fake_st.cache_data = lambda **kw: (lambda f: f)
        fake_st.cache_resource = lambda **kw: (lambda f: f)
        fake_st.set_page_config = lambda **kw: None
        fake_st.secrets = secrets or {}

        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None

        monkeypatch.setitem(sys.modules, "streamlit", fake_st)
        monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

        if env_val is not None:
            monkeypatch.setenv("TMDB_API_KEY", env_val)
        else:
            monkeypatch.delenv("TMDB_API_KEY", raising=False)

        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        return importlib.import_module("app")

    def test_secrets_takes_priority(self, monkeypatch):
        mod = self._load_app_module(
            monkeypatch,
            secrets={"TMDB_API_KEY": "secret_key"},
            env_val="env_key",
        )
        assert mod.TMDB_API_KEY == "secret_key"

    def test_env_fallback_when_no_secrets(self, monkeypatch):
        mod = self._load_app_module(
            monkeypatch,
            secrets={},
            env_val="env_key",
        )
        assert mod.TMDB_API_KEY == "env_key"

    def test_no_key_gives_empty_string(self, monkeypatch):
        mod = self._load_app_module(monkeypatch, secrets={}, env_val=None)
        assert mod.TMDB_API_KEY == ""

    def test_placeholder_key_treated_as_empty(self, monkeypatch):
        mod = self._load_app_module(
            monkeypatch,
            secrets={},
            env_val="your_tmdb_api_key_here",
        )
        assert mod.TMDB_API_KEY == ""
