# jev-on-durians

Tutorial + benchmark: decision models (Jev, Kev and co.) vs LLMs on 83 durian opinions; the results
feed a szia.ai article. Spec: `instructions/GOAL_SPEC.md` (temporary folder).

- Python 3.14 in `.venv` (uv). Install with `uv pip install -e ".[dev]"`; never create `uv.lock`.
- ruff via pre-commit for lint and format; `pytest -q` runs offline.
- The notebook is the tutorial (visible API calls); `src/` holds the machinery (runner, `.cache/`,
  metrics, costs, Elo, charts, README writer).
- Keys are in `.env`: load them, never print, log or commit them; `.env.example` mirrors the names.
- `data/durian_opinions.csv` is read-only; Marton fixes labels himself.
- Never `git commit` or `git push` - Marton commits.
- Minimal modifications: change only what the task needs; keep existing files and names.
