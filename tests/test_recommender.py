"""Tests on a tiny fake catalog with fake embeddings - no downloads needed."""
import numpy as np
import pandas as pd

from song_recs import Recommender, Request
from song_recs.data import base_title, clean
from song_recs.features import describe


def fake_raw() -> pd.DataFrame:
    rows = []
    for i in range(40):
        calm = i < 20
        rows.append({
            "Unnamed: 0": i, "track_id": f"id{i}", "artists": f"Artist{i % 10}",
            "album_name": "A", "track_name": f"Song {i}", "popularity": 100 - i,
            "duration_ms": 200_000, "explicit": False,
            "danceability": 0.3 if calm else 0.8, "energy": 0.2 if calm else 0.9, "key": 0,
            "loudness": -14.0 if calm else -4.0, "mode": 1, "speechiness": 0.04,
            "acousticness": 0.9 if calm else 0.05, "instrumentalness": 0.0, "liveness": 0.1,
            "valence": 0.2 if calm else 0.8, "tempo": 80.0 if calm else 128.0, "time_signature": 4,
            "track_genre": "acoustic" if calm else "edm",
        })
    rows.append({**rows[0], "track_name": "Song 0 - Remastered 2011", "track_genre": "folk", "popularity": 50})
    return pd.DataFrame(rows)


def make_rec() -> Recommender:
    df = clean(fake_raw(), size=1000)
    df["description"] = df.apply(describe, axis=1)
    E = np.where(df["genre"].eq("acoustic").to_numpy()[:, None], [1.0, 0.0], [0.0, 1.0]).astype(np.float32)
    return Recommender(df, text_emb=E)


def test_base_title():
    assert base_title("Bohemian Rhapsody - Remastered 2011") == "bohemian rhapsody"
    assert base_title("Stay (with Justin Bieber)") == "stay"
    assert base_title("Live Forever") == "live forever"


def test_clean_merges_versions_and_genres():
    df = clean(fake_raw(), size=1000)
    assert len(df) == 40
    assert df.loc[df["track_name"] == "Song 0", "genres"].iloc[0] == ["acoustic", "folk"]


def test_describe_uses_words():
    d = describe(pd.Series({"tempo": 80, "energy": 0.2, "valence": 0.1, "danceability": 0.3,
                            "acousticness": 0.9, "instrumentalness": 0, "speechiness": 0.03,
                            "liveness": 0.1, "loudness": -12, "genres": ["indie-pop"]}))
    assert d.startswith("indie pop song.") and "sad" in d and "acoustic" in d


def test_recommends_similar_sound_and_respects_artist_cap():
    rec = make_rec()
    calm_seed = int(rec.df.index[rec.df["genre"] == "acoustic"][0])
    out = rec.recommend(Request(liked=[calm_seed], k=6, text_weight=0.0, max_per_artist=1))
    assert (out["genre"] == "acoustic").all()
    assert out["artists"].is_unique
    assert calm_seed not in out.index


def test_skips_push_away():
    rec = make_rec()
    calm = rec.df.index[rec.df["genre"] == "acoustic"].tolist()
    loud = rec.df.index[rec.df["genre"] == "edm"].tolist()
    out = rec.recommend(Request(liked=[calm[0], loud[0]], skipped=loud[1:4], k=5, text_weight=0.0))
    assert (out["genre"] == "acoustic").mean() >= 0.6
