# Taste Engine — Phase 1 (headless analysis core)

The defensible spine of the music-taste personality scanner. **No LLM, no
Spotify calls in this layer** — just explainable math that turns listening
data into a structured profile. The LLM narrator (Phase 2) and the Spotify
input adapters (Phase 3) sit *on top of* this; they never replace it.

## Why this exists first

The whole project lives or dies on one interview question: *"how does it
actually work?"* If the answer is "I prompted GPT," it's a wrapper. This layer
is the answer that survives scrutiny — every output number has a formula you
can explain in a sentence.

## Run it

```bash
pip install pytest
python analyze.py sample_data.json   # see two contrasting users + compatibility
python -m pytest tests/ -v           # 19 tests covering the math + invariants
```

## The intermediate shape

Both Spotify input paths — `/me/top/artists` (solo) and a pasted playlist URL
(comparison) — converge to the **same** list of `ArtistEntry` objects. The
engine never knows which path produced its input. That convergence keeps the
two future adapters thin and the analysis single-sourced.

```
ArtistEntry(name, genres[], popularity, release_year, weight)
```

## What it computes

| Output | Formula | Meaning |
|---|---|---|
| `genre_vector` | weighted genre frequency, normalized to sum=1 | the taste fingerprint (cosine / clustering input) |
| `mainstream` | weighted mean Spotify popularity / 100 | obscure ↔ chart pop |
| `nostalgia` | weighted-median release-year age / 40yr window | current ↔ classic |
| `eclectic` | Shannon entropy of genre vector / log2(n) | focused ↔ spread |
| `explorer` | inverse-Simpson effective artist count, normalized | same-few ↔ wide spread |
| `compatibility` | 0.65·cosine(genres) + 0.35·axis closeness | why two people match |

## Known knob (deliberate, not a bug)

`eclectic` is **pure normalized entropy**, so it measures spread *relative to
how many genres are present*. A 3-genre user can still score high if those
three are evenly weighted. If you want raw genre count to matter, blend a
saturation curve on `distinct_genres`. Left as-is for defensibility; revisit
once real data is in.

## Where this goes next

- **Phase 2** — wrap `analyze()` in FastAPI, add LLM narrator (numbers in → roast out)
- **Phase 3** — Next.js frontend + Spotify OAuth, both input adapters
- **Phase 4** — log every `genre_vector` to Supabase; ship compatibility mode
- **Phase 5** — once the corpus is large enough, K-means on logged vectors to
  *discover* archetypes, validated against these hand-defined axes
