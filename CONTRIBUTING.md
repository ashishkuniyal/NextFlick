# Contributing

Thank you for considering a contribution to **NextFlick**! Here's everything you need to know to get started.

---

## Development setup

```bash
git clone https://github.com/ashishkuniyal/Movie-Recommender-System.git
cd Movie-Recommender-System

python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# Unix:     source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## Ground rules

- **Never commit secrets** — use `.env` locally or Streamlit Secrets on Cloud.
- **Never commit the raw CSVs** — they are gitignored; the compiled `.pkl` files are committed instead.
- **Keep `app.py` at the repo root** — Streamlit Cloud is configured to read it from there.
- **Keep `requirements.txt` runtime-only** — heavy dev/build deps go in `requirements-dev.txt`.

---

## Branching workflow

1. Create a feature branch off `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. Make your changes and run the test suite:
   ```bash
   pytest tests/ -v
   ```
3. Open a pull request against `main`.
4. Merging to `main` triggers an automatic redeploy on Streamlit Community Cloud.

---

## Running tests

```bash
pytest tests/ -v
```

Tests are fully offline — no network calls or Streamlit server required.

---

## Reporting bugs or ideas

Open a GitHub Issue with:
- A clear title
- Steps to reproduce (for bugs)
- Expected vs. actual behaviour
- Python and package versions (`pip freeze`)
