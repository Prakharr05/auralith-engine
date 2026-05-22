"""
The taste analysis engine.

This is the spine of the project — the part that turns raw listening data into
a structured, *defensible* profile. There is deliberately NO LLM in this file.
Every number that comes out of here is computed with an explainable formula:

  - genre_vector : weighted, normalized genre frequency distribution
  - mainstream   : weighted mean of Spotify popularity (0-100) -> [0,1]
  - nostalgia    : how far the listening skews toward older release years
  - eclectic     : Shannon entropy of the genre distribution, normalized
  - explorer     : artist spread (effective number of artists vs. total weight)

The LLM only ever sees the *output* of this file and writes prose around it.
That separation is the whole pitch: "analysis engine + presentation layer,"
not "I prompted a model."
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median

from .models import ArtistEntry, TasteProfile

# Reference point for the nostalgia axis. We measure how old the music is
# relative to "now"; NOSTALGIA_WINDOW sets how many years back counts as
# "maximally nostalgic" so the score saturates instead of growing forever.
NOSTALGIA_WINDOW = 40  # a 40-year-old median => nostalgia score near 1.0


def build_genre_vector(artists: list[ArtistEntry]) -> dict[str, float]:
    """
    Build the taste vector: a genre -> weight map normalized to sum to 1.0.

    Each artist contributes its `weight`, split evenly across the genres tagged
    on it. An artist with no genres contributes nothing to the vector (it still
    counts for the artist-spread axis). Splitting weight across genres means a
    one-genre artist concentrates its mass while a five-genre artist spreads it,
    which is exactly the behavior we want for the entropy measure downstream.
    """
    accum: dict[str, float] = defaultdict(float)
    for a in artists:
        if not a.genres:
            continue
        share = a.weight / len(a.genres)
        for g in a.genres:
            accum[g] += share

    total = sum(accum.values())
    if total == 0:
        return {}
    return {g: w / total for g, w in accum.items()}


def shannon_entropy_bits(distribution: dict[str, float]) -> float:
    """
    Shannon entropy of a probability distribution, in bits.

        H = -sum(p * log2(p))

    Higher entropy => weight spread across many genres (eclectic taste).
    Lower entropy => weight concentrated in a few genres (focused taste).
    This is the citable formula behind the 'eclectic' axis.
    """
    h = 0.0
    for p in distribution.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


def eclectic_score(genre_vector: dict[str, float]) -> float:
    """
    Normalize entropy into [0, 1] by dividing by the maximum possible entropy
    for the number of genres present (log2(n)). This makes the score comparable
    across users with different numbers of genres: 1.0 means weight is spread
    perfectly evenly, 0.0 means it's all in a single genre.
    """
    n = len(genre_vector)
    if n <= 1:
        return 0.0
    h = shannon_entropy_bits(genre_vector)
    max_h = math.log2(n)
    return h / max_h if max_h > 0 else 0.0


def mainstream_score(artists: list[ArtistEntry]) -> float:
    """
    Weighted mean of Spotify popularity (0-100), scaled to [0, 1].
    1.0 = exclusively chart-topping artists; 0.0 = exclusively obscure ones.
    """
    total_w = sum(a.weight for a in artists)
    if total_w == 0:
        return 0.0
    weighted = sum(a.popularity * a.weight for a in artists)
    return (weighted / total_w) / 100.0


def nostalgia_score(artists: list[ArtistEntry], now_year: int | None = None) -> float:
    """
    How far the listening skews toward older music, in [0, 1].

    We take the weighted median release year, measure its age relative to the
    current year, and saturate at NOSTALGIA_WINDOW. Using the median (not mean)
    keeps one ancient track from dominating, and ignores artists with no year.
    """
    if now_year is None:
        now_year = datetime.now(timezone.utc).year

    years: list[int] = []
    weights: list[float] = []
    for a in artists:
        if a.release_year is not None:
            years.append(a.release_year)
            weights.append(a.weight)

    if not years:
        return 0.0

    med_year = _weighted_median(years, weights)
    age = max(0, now_year - med_year)
    return min(1.0, age / NOSTALGIA_WINDOW)


def explorer_score(artists: list[ArtistEntry]) -> float:
    """
    Artist spread, in [0, 1]. Uses the 'effective number of artists' idea:
    if all weight sits on one artist the score is ~0 (you replay the same few);
    if weight is spread evenly across many artists it approaches 1 (explorer).

    Computed as the inverse Simpson index normalized by artist count:
        effective_n = 1 / sum(p_i^2)     where p_i = artist_weight / total
        score       = (effective_n - 1) / (n - 1)
    """
    n = len(artists)
    if n <= 1:
        return 0.0
    total_w = sum(a.weight for a in artists)
    if total_w == 0:
        return 0.0
    shares = [a.weight / total_w for a in artists]
    simpson = sum(p * p for p in shares)
    effective_n = 1.0 / simpson
    return (effective_n - 1) / (n - 1)


def _weighted_median(values: list[int], weights: list[float]) -> int:
    """Weighted median: smallest value where cumulative weight crosses half."""
    paired = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(weights)
    if total == 0:
        return int(median(values))
    half = total / 2
    cum = 0.0
    for v, w in paired:
        cum += w
        if cum >= half:
            return v
    return paired[-1][0]


def analyze(artists: list[ArtistEntry], now_year: int | None = None,
            top_n: int = 8) -> TasteProfile:
    """
    Run the full pipeline and return a TasteProfile. This is the single entry
    point the FastAPI service calls. Input is the converged intermediate shape;
    output is the structured profile the narrator will voice.
    """
    if not artists:
        raise ValueError("analyze() requires at least one ArtistEntry")

    genre_vector = build_genre_vector(artists)
    raw_entropy = shannon_entropy_bits(genre_vector)

    top_genres = sorted(genre_vector.items(), key=lambda x: x[1], reverse=True)[:top_n]

    years = [a.release_year for a in artists if a.release_year is not None]
    med_year = int(median(years)) if years else None

    return TasteProfile(
        genre_vector=genre_vector,
        mainstream_score=mainstream_score(artists),
        nostalgia_score=nostalgia_score(artists, now_year=now_year),
        eclectic_score=eclectic_score(genre_vector),
        explorer_score=explorer_score(artists),
        top_genres=top_genres,
        distinct_genres=len(genre_vector),
        distinct_artists=len(artists),
        median_release_year=med_year,
        raw_entropy_bits=raw_entropy,
    )


def compatibility(profile_a: TasteProfile, profile_b: TasteProfile) -> dict:
    """
    Compatibility between two users (the Phase 4 hook, built now because it's
    cheap and it's the viral mechanic). Combines:

      - cosine similarity over the shared genre-vector space (taste overlap)
      - closeness on the four scalar axes (vibe alignment)

    Returns the components separately so the UI/narrator can say *why* two
    people match, not just a single opaque number.
    """
    genre_sim = _cosine(profile_a.genre_vector, profile_b.genre_vector)

    axes_a = [profile_a.mainstream_score, profile_a.nostalgia_score,
              profile_a.eclectic_score, profile_a.explorer_score]
    axes_b = [profile_b.mainstream_score, profile_b.nostalgia_score,
              profile_b.eclectic_score, profile_b.explorer_score]
    # Axis closeness: 1 - mean absolute distance across the four axes.
    axis_closeness = 1.0 - sum(abs(x - y) for x, y in zip(axes_a, axes_b)) / 4.0

    # Blend: genre overlap weighted a bit more than vibe, since shared genres
    # are the stronger signal of actual taste compatibility.
    overall = 0.65 * genre_sim + 0.35 * axis_closeness

    return {
        "overall": round(overall, 4),
        "genre_similarity": round(genre_sim, 4),
        "axis_closeness": round(axis_closeness, 4),
    }


def _cosine(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity over two sparse genre vectors keyed by genre name."""
    shared = set(vec_a) & set(vec_b)
    if not shared:
        return 0.0
    dot = sum(vec_a[g] * vec_b[g] for g in shared)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
