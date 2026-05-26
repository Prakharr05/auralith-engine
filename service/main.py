"""
FastAPI service — the intelligence layer as one deployable unit.

Endpoints:
  POST /analyze            full profile + roast from the converged shape
  POST /analyze/top        same, from a raw /me/top/artists payload (Path A)
  POST /analyze/playlist   same, from raw playlist + artists payloads (Path B)
  POST /compatibility      two profiles -> compatibility breakdown
  GET  /health             liveness

The Next.js frontend does OAuth + Spotify fetches, then POSTs the raw payloads
here. This service owns ALL the intelligence (engine + narrator) so it's the
single thing you describe as "the analysis engine" — Next is just the face.
"""

from __future__ import annotations

# Load .env BEFORE importing anything that reads env vars at import time.
# Without this, uvicorn does not pick up .env and the narrator silently falls
# back to the deterministic template. load_dotenv() is a no-op if python-dotenv
# isn't installed or no .env exists, so it's safe everywhere.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from typing import Any, Optional
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine import ArtistEntry, analyze, compatibility
from service.adapters import from_playlist, from_top_artists
from service.narrator import narrate

app = FastAPI(title="Music Taste Personality Engine", version="0.2.0")

# CORS so the Next.js frontend (different origin) can call us. In production set
# ALLOWED_ORIGINS to your Vercel URL (comma-separated for multiple); defaults to
# "*" for local dev convenience.
_origins = os.getenv("ALLOWED_ORIGINS", "*")
allow_origins = ["*"] if _origins.strip() == "*" else [o.strip() for o in _origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---- request/response models ---------------------------------------------

class ArtistInput(BaseModel):
    name: str
    genres: list[str] = Field(default_factory=list)
    popularity: int = 0
    release_year: Optional[int] = None
    weight: float = 1.0


class AnalyzeRequest(BaseModel):
    artists: list[ArtistInput]


class TopArtistsRequest(BaseModel):
    payload: dict[str, Any]  # raw /me/top/artists JSON


class PlaylistRequest(BaseModel):
    tracks: dict[str, Any]   # raw /playlists/{id}/tracks JSON
    artists: dict[str, Any]  # raw /artists?ids=... JSON


class CompatibilityRequest(BaseModel):
    artists_a: list[ArtistInput]
    artists_b: list[ArtistInput]


# ---- helpers --------------------------------------------------------------

def _to_entries(artists: list[ArtistInput]) -> list[ArtistEntry]:
    return [ArtistEntry(**a.model_dump()) for a in artists]


def _profile_and_roast(entries: list[ArtistEntry]) -> dict[str, Any]:
    if not entries:
        raise HTTPException(status_code=422, detail="No artists with usable data.")
    profile = analyze(entries)
    # Pass artist names (heaviest-weighted first) so the narrator can ground the
    # read in who the person actually listens to — essential now that Spotify
    # frequently returns empty genres.
    top_artists = [e.name for e in sorted(entries, key=lambda e: e.weight, reverse=True)]
    return {"profile": profile.as_dict(), "narration": narrate(profile, top_artists)}


# ---- endpoints ------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze_endpoint(req: AnalyzeRequest) -> dict[str, Any]:
    """Analyze the already-converged intermediate shape."""
    return _profile_and_roast(_to_entries(req.artists))


@app.post("/analyze/top")
def analyze_top(req: TopArtistsRequest) -> dict[str, Any]:
    """Path A: raw /me/top/artists payload."""
    return _profile_and_roast(from_top_artists(req.payload))


@app.post("/analyze/playlist")
def analyze_playlist(req: PlaylistRequest) -> dict[str, Any]:
    """Path B: raw playlist tracks + batched artists payloads."""
    return _profile_and_roast(from_playlist(req.tracks, req.artists))


@app.post("/compatibility")
def compatibility_endpoint(req: CompatibilityRequest) -> dict[str, Any]:
    """Two users -> compatibility breakdown + both profiles."""
    entries_a = _to_entries(req.artists_a)
    entries_b = _to_entries(req.artists_b)
    if not entries_a or not entries_b:
        raise HTTPException(status_code=422, detail="Both users need artists.")
    profile_a = analyze(entries_a)
    profile_b = analyze(entries_b)
    return {
        "compatibility": compatibility(profile_a, profile_b),
        "profile_a": profile_a.as_dict(),
        "profile_b": profile_b.as_dict(),
    }
