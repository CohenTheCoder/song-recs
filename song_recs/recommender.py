"""Step 3: recommend.

    taste      = average of the songs you like  -  0.5 x average of the songs you skip
                 (the Rocchio update from information retrieval; works in both vector spaces)
    relevance  = w_audio * cosine(audio taste, song) + w_text * cosine(text taste or vibe query, song)
    re-rank    = Maximal Marginal Relevance: repeatedly pick the song with the best
                 lambda * relevance - (1 - lambda) * sound-similarity-to-songs-already-picked,
                 so you don't get ten near-identical tracks. Max N songs per artist.

Every recommendation comes with a reason ("similar energy and mood to your likes", "matches
'rainy day'"), because explainable recs earn more trust than black boxes.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .data import load_catalog
from .features import audio_matrix, describe, embed, text_matrix

READABLE = {"danceability": "danceability", "energy": "energy", "valence": "mood (valence)",
            "acousticness": "acousticness", "tempo": "tempo"}


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@dataclass
class Request:
    liked: list[int] = field(default_factory=list)     # catalog row indices
    skipped: list[int] = field(default_factory=list)
    vibe: str = ""                                     # free-text description
    k: int = 10
    text_weight: float = 0.5                           # 0 = audio only, 1 = text only
    diversity: float = 0.2                             # 0 = pure relevance, 1 = max variety
    max_per_artist: int = 2
    popularity: tuple[int, int] = (0, 100)
    popularity_boost: float = 0.15                     # nudge toward songs people know


class Recommender:
    def __init__(self, catalog: pd.DataFrame | None = None, genre_weight: float = 0.6,
                 text_emb: np.ndarray | None = None):
        self.df = load_catalog() if catalog is None else catalog
        if "description" not in self.df:
            self.df["description"] = self.df.apply(describe, axis=1)
        self.A = audio_matrix(self.df, genre_weight)
        self.S = audio_matrix(self.df, genre_weight=0.0)  # pure sound, used for variety
        self.E = text_emb if text_emb is not None else text_matrix(self.df)
        # cap by PRIMARY artist so "X" and "X, Featured Guest" count as the same artist
        self._artists = self.df["artists"].str.split(", ").str[0].to_numpy()

    # ---------- search ----------
    def search(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Find a song by title/artist to use as a seed."""
        q = query.lower().strip()
        hits = self.df[self.df["label"].str.lower().str.contains(q, regex=False)]
        return hits.nlargest(limit, "popularity")

    # ---------- core ----------
    def relevance(self, req: Request) -> tuple[np.ndarray, dict]:
        parts = {}
        rel = np.zeros(len(self.df), dtype=np.float32)
        w_text = req.text_weight if (req.vibe or req.liked) else 0.0
        if req.liked:
            taste_a = self.A[req.liked].mean(0)
            if req.skipped:
                taste_a = taste_a - 0.5 * self.A[req.skipped].mean(0)
            parts["audio"] = self.A @ _unit(taste_a)
            rel += (1 - w_text) * parts["audio"]
        text_vecs = []
        if req.liked:
            t = self.E[req.liked].mean(0)
            if req.skipped:
                t = t - 0.5 * self.E[req.skipped].mean(0)
            text_vecs.append(_unit(t))
        if req.vibe:
            text_vecs.append(embed([req.vibe])[0])
        if text_vecs:
            parts["text"] = self.E @ _unit(np.mean(text_vecs, axis=0))
            rel += (w_text if req.liked else 1.0) * parts["text"]
        pop = self.df["popularity"].to_numpy(dtype=np.float32) / 100
        if not req.liked and not req.vibe:  # cold start: popular songs
            return pop, parts
        rel = (1 - req.popularity_boost) * rel + req.popularity_boost * pop
        return rel, parts

    def recommend(self, req: Request) -> pd.DataFrame:
        rel, parts = self.relevance(req)
        lo, hi = req.popularity
        pop = self.df["popularity"].to_numpy()
        mask = (pop >= lo) & (pop <= hi)
        mask[req.liked + req.skipped] = False
        seen = set(self.df["song_key"].iloc[req.liked + req.skipped])
        mask &= ~self.df["song_key"].isin(seen).to_numpy()  # no other versions of seeds
        cand = np.where(mask)[0]
        cand = cand[np.argsort(-rel[cand])][:400]  # shortlist, then re-rank for variety

        chosen, per_artist = [], {}
        lam = 1 - req.diversity
        # rescale the shortlist's relevance to 0..1 so it's on the same footing as similarity
        r = rel[cand] if cand.size else rel[:0]
        norm = {int(i): float(v) for i, v in zip(cand, (r - r.min()) / (np.ptp(r) + 1e-9))}
        while cand.size and len(chosen) < req.k:
            relc = np.array([norm[int(i)] for i in cand])
            if chosen:
                # redundancy is measured on SOUND only, so variety means "different-sounding
                # songs that still fit", not "drift into other genres"
                redundancy = (self.S[cand] @ self.S[chosen].T).max(axis=1)
                mmr = lam * relc - (1 - lam) * redundancy
            else:
                mmr = relc
            for j in np.argsort(-mmr):
                i = int(cand[j])
                if per_artist.get(self._artists[i], 0) < req.max_per_artist:
                    chosen.append(i)
                    per_artist[self._artists[i]] = per_artist.get(self._artists[i], 0) + 1
                    break
            cand = cand[~np.isin(cand, chosen)]
            cand = cand[[per_artist.get(self._artists[i], 0) < req.max_per_artist for i in cand]]

        out = self.df.iloc[chosen].copy()
        out["match"] = rel[chosen]
        out["why"] = [self.explain(i, req, parts) for i in chosen]
        return out

    def explain(self, i: int, req: Request, parts: dict) -> str:
        row = self.df.iloc[i]
        reasons = []
        if req.liked:
            liked = self.df.iloc[req.liked]
            close = [READABLE[f] for f in READABLE
                     if abs(row[f] - liked[f].mean()) <= 0.08 * (1 if f != "tempo" else 150)]
            if close:
                reasons.append("similar " + ", ".join(close[:3]))
            shared = set(row["genres"]) & {g for gs in liked["genres"] for g in gs}
            if shared:
                reasons.append("genre: " + ", ".join(sorted(shared)[:2]))
            if row["artists"] in set(liked["artists"]):
                reasons.append("artist you liked")
        if req.vibe and "text" in parts:
            reasons.append(f"fits “{req.vibe}” ({parts['text'][i]:.2f})")
        return "; ".join(reasons) or "popular pick"

    def taste_profile(self, idx: list[int]) -> pd.Series:
        return self.df.iloc[idx][["danceability", "energy", "valence", "acousticness", "instrumentalness",
                                  "speechiness", "liveness"]].mean()


def build_all(show_progress: bool = True) -> Recommender:
    """Download + clean + embed everything (run once via scripts/build_index.py)."""
    df = load_catalog()
    df["description"] = df.apply(describe, axis=1)
    E = text_matrix(df, show_progress)
    return Recommender(df, text_emb=E)


__all__ = ["Recommender", "Request", "build_all", "config"]
