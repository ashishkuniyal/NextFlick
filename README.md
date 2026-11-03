# 🎬 NextFlick — Movie Recommender System

<div align="center">

[![Live Demo](https://img.shields.io/badge/▶_Live_Demo-nextflick--ai.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://nextflick-ai.streamlit.app)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

**A content-based movie recommendation engine built with NLP, Bag-of-Words vectorization, and Cosine Similarity — deployed live on Streamlit Community Cloud.**

[**Try the live demo →**](https://nextflick-ai.streamlit.app)

</div>

---

## ✨ What it does

Pick any of ~4,800 movies and NextFlick instantly suggests the 3–10 most similar titles by analysing five content signals in parallel:

| Signal | Detail |
|--------|--------|
| 📖 **Plot Overview** | Key semantic words from the synopsis |
| 🎭 **Genres** | e.g. Action, Sci-Fi, Thriller |
| 🏷️ **Keywords** | Curated plot tags (e.g. *space-travel*, *superhero*) |
| 🎬 **Cast** | Top 3 billed actors |
| 🎥 **Director** | Credited director |

All five signals are merged into a single "tags" string per movie, vectorized with `CountVectorizer` (5,000 features), and compared pairwise using Cosine Similarity. The precomputed 4,800 × 4,800 matrix (`float16`, ~44 MB) is committed to Git and loaded once at startup via `@st.cache_resource`.

---

## 🏗️ Architecture

```
 tmdb_5000_movies.csv ──┐
                        ├─► Merge on (id, title) ──► Feature Extraction
 tmdb_5000_credits.csv ─┘         │
                                   ▼
                        genres + keywords + cast[:3]
                        + director + overview tokens
                                   │
                                   ▼
                         Space Collapse  ──►  Porter Stemming
                         ("Sam Worthington" → "SamWorthington")
                                   │
                                   ▼
                        CountVectorizer(max_features=5000)
                                   │
                                   ▼
                        Cosine Similarity  (4800 × 4800, float16)
                                   │
                        ┌──────────┴──────────┐
                        ▼                     ▼
                 movie_list.pkl         similarity.pkl
                        │                     │
                        └──────────┬──────────┘
                                   ▼
                          Streamlit Web App
                    ┌──────────────────────────┐
                    │  Searchable dropdown      │
                    │  🎲 Surprise me button    │
                    │  Slider: 3–10 results     │
                    │  TMDB poster art          │
                    │  Match % + TMDB links     │
                    └──────────────────────────┘
```

### The maths

Text is mapped to a 5,000-dimensional Bag-of-Words space. Similarity between movies $A$ and $B$ is:

$$\text{Similarity}(A, B) = \cos\theta = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\|\,\|\vec{B}\|}$$

Scores range from `0.0` (no overlap) to `1.0` (identical tags).

---

## 📁 Project Structure

```
.
├── app.py                  # Streamlit entrypoint (Streamlit Cloud reads this)
├── generate_model.py       # CLI to rebuild model/ artifacts from raw CSVs
├── requirements.txt        # Runtime deps — pinned (Streamlit Cloud reads this)
├── requirements-dev.txt    # Dev/test deps — NOT installed on Cloud
├── .env.example            # API key template (safe to commit)
├── .streamlit/
│   └── config.toml         # App theme & server settings
├── model/
│   ├── movie_list.pkl      # Preprocessed movie DataFrame (~2 MB, committed)
│   └── similarity.pkl      # Cosine similarity matrix (~44 MB, committed)
├── data/
│   ├── tmdb_5000_movies.csv   # Raw dataset (gitignored — download separately)
│   └── tmdb_5000_credits.csv  # Raw dataset (gitignored — download separately)
├── tests/
│   └── test_smoke.py       # Offline pytest smoke tests
└── notebook86c26b4f17.ipynb   # Exploratory analysis notebook
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10+
- Git

### 1. Clone

```bash
git clone https://github.com/ashishkuniyal/Movie-Recommender-System.git
cd Movie-Recommender-System
```

### 2. Create a virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure TMDB API key *(optional — the app works without it)*

```bash
cp .env.example .env
# Open .env and paste your key:
# TMDB_API_KEY=your_actual_key_here
```

Get a free key at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api).

### 5. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **Model files included.** `model/movie_list.pkl` and `model/similarity.pkl` are committed to the repo (~46 MB total), so the app works out-of-the-box after cloning — no dataset download needed.

---

## 🔨 Rebuilding the Model

If you want to regenerate the artifacts from the raw CSVs:

1. Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle.
2. Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` into `data/`.
3. Run:

```bash
python generate_model.py
```

Completes in ~13 seconds and writes `model/movie_list.pkl` + `model/similarity.pkl`.

---

## 🧪 Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

The test suite runs fully offline — no network calls, no Streamlit server needed.

---

## ☁️ Streamlit Cloud Deployment

The app is deployed at **[nextflick-ai.streamlit.app](https://nextflick-ai.streamlit.app)**.

**To add your TMDB API key on Streamlit Cloud:**
1. Go to your app dashboard → **Settings → Secrets**.
2. Add:
   ```toml
   TMDB_API_KEY = "your_actual_key_here"
   ```
3. Save — the app rereads secrets on the next request.

The app reads `st.secrets` first, then `os.getenv("TMDB_API_KEY")` as a fallback. If neither is set, it displays SVG placeholder posters and continues working normally.

---

## 🐛 Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| **IndexError: index out of bounds** | `dropna()` left gaps at indices `[2658, 4145, 4437]`; matrix has rows `0..4799` | Added `.reset_index(drop=True)` after `dropna()` |
| **Silent recommendation drift** | After the index gap, every subsequent movie queried the wrong similarity row | Re-alignment via reset index |
| **Cross-contaminated duplicate rows** | `merge(on='title')` cross-joined movies with the same title across years | Changed to `merge(left_on=['id','title'], right_on=['movie_id','title'])` |
| **UI freeze on every rerun** | 185 MB pkl loaded at top-level on every Streamlit rerun | Wrapped in `@st.cache_resource` |
| **Crash on missing models** | Unhandled `FileNotFoundError` on fresh clone | Friendly in-app error with setup instructions |
| **Hardcoded secret encouragement** | UI told users to put the API key inside `app.py` | Key loaded from `st.secrets` → `.env` chain |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI & App | [Streamlit](https://streamlit.io/) |
| Data wrangling | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| NLP & ML | [scikit-learn](https://scikit-learn.org/) (`CountVectorizer`, `cosine_similarity`), [NLTK](https://www.nltk.org/) (`PorterStemmer`) |
| Poster API | [TMDB API v3](https://developers.themoviedb.org/3) via [Requests](https://requests.readthedocs.io/) |
| Secrets | [python-dotenv](https://github.com/theskumar/python-dotenv) + Streamlit Secrets |
| Testing | [pytest](https://pytest.org/) |

---

## 📄 License & Attribution

Released under the [MIT License](LICENSE).

- Movie metadata and poster images: [The Movie Database (TMDB)](https://www.themoviedb.org/)
- Dataset: [Kaggle TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
