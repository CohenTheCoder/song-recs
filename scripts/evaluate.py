"""Offline evaluation: how good are the recommendations, and what are the trade-offs?

We have no real listening history, so we use a standard proxy task:
  "Given one seed song, do the top-10 recommendations share its genre?"
To keep that fair, the audio model here is built WITHOUT genre tags (genre_weight=0), so it
must find same-genre songs from sound alone.

Metrics
- genre precision@10: share of recs that share a genre tag with the seed (accuracy proxy)
- intra-list diversity: 1 - avg cosine similarity between recs (higher = more varied list)
- unique recs: distinct songs recommended / total recommendations (low = "rich get richer")
- avg popularity: popularity bias. Do we only recommend hits?

Models compared: random · most-popular · audio kNN · audio kNN + MMR diversity re-ranking.

Usage:  python scripts/evaluate.py --seeds 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from song_recs import Recommender, Request  # noqa: E402
from song_recs.features import audio_matrix  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rec = Recommender(genre_weight=0.0)  # no genre leakage
    df, k = rec.df, args.k
    A = audio_matrix(df, genre_weight=0.0)
    genres = df["genres"].map(set).to_numpy()
    rng = np.random.default_rng(0)
    seeds = rng.choice(len(df), args.seeds, replace=False)
    top_pop = np.argsort(-df["popularity"].to_numpy())

    def no_text(req: Request) -> Request:
        req.text_weight, req.popularity_boost = 0.0, 0.0
        return req

    systems = {
        "random": lambda s: rng.choice(len(df), k, replace=False),
        "most popular": lambda s: [i for i in top_pop[: k + 1] if i != s][:k],
        "audio kNN": lambda s: rec.recommend(no_text(Request(liked=[int(s)], k=k, diversity=0.0,
                                                              max_per_artist=k))).index.to_numpy(),
        "audio kNN + MMR (0.2)": lambda s: rec.recommend(no_text(Request(liked=[int(s)], k=k,
                                                                         diversity=0.2))).index.to_numpy(),
        "audio kNN + MMR (0.5)": lambda s: rec.recommend(no_text(Request(liked=[int(s)], k=k,
                                                                         diversity=0.5))).index.to_numpy(),
    }
    rows = []
    for name, fn in systems.items():
        prec, div, pop, seen = [], [], [], set()
        for s in seeds:
            recs = np.asarray(fn(s))
            prec.append(np.mean([bool(genres[i] & genres[s]) for i in recs]))
            sims = A[recs] @ A[recs].T
            div.append(1 - sims[np.triu_indices(len(recs), 1)].mean())
            pop.append(df["popularity"].iloc[recs].mean())
            seen.update(recs.tolist())
        rows.append({"model": name, f"genre_precision@{k}": np.mean(prec), "diversity": np.mean(div),
                     "unique_recs": len(seen) / (len(seeds) * k), "avg_popularity": np.mean(pop)})
        print(f"  {name:24s} done")
    table = pd.DataFrame(rows).round(3)
    print("\n" + table.to_string(index=False))
    out = ROOT / "docs/results.md"
    out.write_text(f"# Offline evaluation ({args.seeds} random seed songs, top-{k})\n\n"
                   + table.to_markdown(index=False) + "\n\n"
                   f"Catalog average popularity: {df['popularity'].mean():.1f}. "
                   "Audio models are built without genre tags, so genre precision measures whether "
                   "sound alone finds same-genre music.\n")
    print(f"\nSaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
