# Running the service

## Install
```bash
pip install -r requirements.txt
```

## Run the API (uses fallback narrator with no key set)
```bash
uvicorn service.main:app --reload --port 8000
```
Interactive docs at http://localhost:8000/docs

## Enable the real LLM narrator
Copy `.env.example` to `.env`, uncomment a provider block, add your key.
The narrator auto-detects it; with no key it uses the deterministic fallback,
so the whole service runs offline.

## Endpoints
- `POST /analyze`          — converged shape `{"artists":[...]}`
- `POST /analyze/top`      — raw `/me/top/artists` payload (Path A)
- `POST /analyze/playlist` — raw playlist tracks + artists payloads (Path B)
- `POST /compatibility`    — two artist lists -> match breakdown
- `GET  /health`

## Test
```bash
python -m pytest tests/ -q    # 29 tests, no key/network needed
```

## The integrity rule (enforced in code)
The narrator is built only from the *computed profile* (`service/narrator.py`,
`build_user_prompt`). Artist and track names never reach the LLM — there's a
test (`test_narrator_prompt_excludes_artist_names`) that fails if they ever do.
This is what keeps it "analysis engine + presentation layer," not a wrapper.
