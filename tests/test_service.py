"""
Phase 2 tests: adapters convert Spotify shapes correctly, the narrator honors
its contract (numbers in -> structured roast out), and the API endpoints wire
everything together. Uses the deterministic fallback narrator so no API key or
network is needed in CI.
"""

from fastapi.testclient import TestClient

from engine import analyze
from service.adapters import collect_artist_ids, from_playlist, from_top_artists
from service.main import app
from service.narrator import build_user_prompt, narrate

client = TestClient(app)


# ---- adapters ------------------------------------------------------------

def test_top_artists_adapter_weights_by_rank():
    payload = {"items": [
        {"name": "First", "genres": ["pop"], "popularity": 90},
        {"name": "Second", "genres": ["rock"], "popularity": 70},
        {"name": "Third", "genres": ["jazz"], "popularity": 50},
    ]}
    entries = from_top_artists(payload)
    assert [e.weight for e in entries] == [3.0, 2.0, 1.0]
    assert entries[0].name == "First"


def test_playlist_adapter_joins_genres_and_counts_tracks():
    tracks = {"items": [
        {"track": {"artists": [{"id": "a1"}], "album": {"release_date": "2019-05-01"}}},
        {"track": {"artists": [{"id": "a1"}], "album": {"release_date": "2021"}}},
        {"track": {"artists": [{"id": "a2"}], "album": {"release_date": "2010-01-01"}}},
    ]}
    artists = {"artists": [
        {"id": "a1", "name": "Alpha", "genres": ["indie"], "popularity": 60},
        {"id": "a2", "name": "Beta", "genres": ["metal"], "popularity": 40},
    ]}
    entries = from_playlist(tracks, artists)
    by_name = {e.name: e for e in entries}
    assert by_name["Alpha"].weight == 2.0      # appears on 2 tracks
    assert by_name["Alpha"].genres == ["indie"]
    assert by_name["Alpha"].release_year == 2019  # earliest of 2019/2021
    assert by_name["Beta"].weight == 1.0


def test_collect_artist_ids_dedupes_in_order():
    tracks = {"items": [
        {"track": {"artists": [{"id": "a1"}, {"id": "a2"}]}},
        {"track": {"artists": [{"id": "a1"}]}},
    ]}
    assert collect_artist_ids(tracks) == ["a1", "a2"]


# ---- narrator contract ---------------------------------------------------

def test_narrator_prompt_includes_artist_names_when_provided():
    # Spotify now often returns empty genres, so we ground the read in artist
    # names. The prompt must include them when passed (still no fabricated facts).
    from engine import ArtistEntry
    entries = [ArtistEntry("Pritam", [], 60, 2015), ArtistEntry("Vishal-Shekhar", [], 55, 2014)]
    prompt = build_user_prompt(analyze(entries), top_artists=["Pritam", "Vishal-Shekhar"])
    assert "Pritam" in prompt
    assert "Vishal-Shekhar" in prompt

def test_narrator_prompt_omits_artist_line_when_none():
    # When no names are passed, no TOP ARTISTS line appears.
    from engine import ArtistEntry
    prompt = build_user_prompt(analyze([ArtistEntry("X", ["pop"], 50, 2020)]))
    assert "TOP ARTISTS" not in prompt


def test_fallback_narrator_returns_full_schema():
    from engine import ArtistEntry
    entries = [ArtistEntry("X", ["pop"], 95, 2023, 5),
               ArtistEntry("Y", ["pop"], 90, 2024, 3)]
    result = narrate(analyze(entries))  # no key set -> fallback
    assert result["_fallback"] is True
    for key in ["personality_type", "summary", "green_flags", "red_flags", "dating_verdict"]:
        assert key in result
    assert len(result["green_flags"]) == 3


# ---- endpoints -----------------------------------------------------------

def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_analyze_endpoint_returns_profile_and_narration():
    body = {"artists": [
        {"name": "A", "genres": ["indie folk"], "popularity": 55, "release_year": 2020, "weight": 3},
        {"name": "B", "genres": ["shoegaze"], "popularity": 48, "release_year": 1994, "weight": 2},
    ]}
    r = client.post("/analyze", json=body)
    assert r.status_code == 200
    data = r.json()
    assert "profile" in data and "narration" in data
    assert set(data["profile"]["axes"]) == {"mainstream", "nostalgia", "eclectic", "explorer"}


def test_analyze_top_endpoint():
    body = {"payload": {"items": [
        {"name": "A", "genres": ["pop"], "popularity": 88},
        {"name": "B", "genres": ["dance pop"], "popularity": 80},
    ]}}
    r = client.post("/analyze/top", json=body)
    assert r.status_code == 200
    assert r.json()["profile"]["distinct_artists"] == 2


def test_analyze_rejects_empty():
    r = client.post("/analyze", json={"artists": []})
    assert r.status_code == 422


def test_compatibility_endpoint():
    body = {
        "artists_a": [{"name": "A", "genres": ["indie folk"], "popularity": 55, "release_year": 2020}],
        "artists_b": [{"name": "B", "genres": ["indie folk"], "popularity": 55, "release_year": 2020}],
    }
    r = client.post("/compatibility", json=body)
    assert r.status_code == 200
    comp = r.json()["compatibility"]
    assert comp["genre_similarity"] == 1.0  # identical genres
