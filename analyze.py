"""
CLI runner for the headless engine.

    python analyze.py sample_data.json

Prints a full TasteProfile for each user in the file, plus a compatibility
score between the first two. No Spotify, no LLM — just the spine, so you can
see and trust the numbers before anything sits on top of them.
"""

import json
import sys
from pathlib import Path

from engine import ArtistEntry, analyze, compatibility


def load_users(path: str) -> dict[str, list[ArtistEntry]]:
    raw = json.loads(Path(path).read_text())
    return {
        user: [ArtistEntry(**entry) for entry in entries]
        for user, entries in raw.items()
    }


def print_profile(name: str, profile) -> None:
    d = profile.as_dict()
    print(f"\n{'=' * 56}")
    print(f"  {name}")
    print(f"{'=' * 56}")
    print(f"  distinct artists : {d['distinct_artists']}")
    print(f"  distinct genres  : {d['distinct_genres']}")
    print(f"  median year      : {d['median_release_year']}")
    print(f"  entropy (bits)   : {d['raw_entropy_bits']}")
    print("\n  AXES")
    for axis, val in d["axes"].items():
        bar = "#" * int(val * 30)
        print(f"    {axis:<11} {val:>6.2f}  {bar}")
    print("\n  TOP GENRES")
    for g in d["top_genres"]:
        print(f"    {g['genre']:<22} {g['weight']:>6.3f}")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_data.json"
    users = load_users(path)

    profiles = {}
    for name, artists in users.items():
        profiles[name] = analyze(artists)
        print_profile(name, profiles[name])

    names = list(profiles)
    if len(names) >= 2:
        a, b = names[0], names[1]
        comp = compatibility(profiles[a], profiles[b])
        print(f"\n{'=' * 56}")
        print(f"  COMPATIBILITY: {a}  <->  {b}")
        print(f"{'=' * 56}")
        print(f"  overall          : {comp['overall']:.2f}")
        print(f"  genre similarity : {comp['genre_similarity']:.2f}")
        print(f"  axis closeness   : {comp['axis_closeness']:.2f}")
    print()


if __name__ == "__main__":
    main()
