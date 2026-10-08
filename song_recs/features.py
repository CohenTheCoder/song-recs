"""Step 2: turn each song into vectors.

Two views of the same song:

1. AUDIO vector: the 9 audio features, standardized (z-scores so tempo's 0-200 range doesn't
   drown out valence's 0-1), plus one-hot genre tags times a weight you control. Unit length,
   so dot product = cosine similarity.

2. TEXT vector: we write a sentence describing the song in words
       "Indie pop song. Slow, calm, mellow, sad, acoustic."
   and embed it with a Hugging Face sentence-transformer. Now you can search the catalog with
   plain English ("upbeat 80s synth for a road trip"), because the query and the songs live in
   the same 384-dimensional meaning space.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from . import config

ROOT = Path(__file__).resolve().parent.parent
EMB_PATH = ROOT / "data/text_embeddings.npy"


def describe(row: pd.Series) -> str:
    """Translate numbers into the words people use for music."""
    words = []
    t = row["tempo"]
    words.append("slow" if t < 90 else "fast" if t > 125 else "mid-tempo")
    e = row["energy"]
    if e > 0.75:
        words += ["high-energy", "intense"]
    elif e < 0.35:
        words += ["calm", "mellow", "chill"]
    v = row["valence"]
    if v > 0.65:
        words += ["happy", "upbeat", "feel-good"]
    elif v < 0.3:
        words += ["sad", "dark", "melancholic"]
    if row["danceability"] > 0.7:
        words += ["danceable", "groovy"]
    if row["acousticness"] > 0.6:
        words += ["acoustic"]
    if row["instrumentalness"] > 0.5:
        words += ["instrumental", "no vocals"]
    if row["speechiness"] > 0.33:
        words += ["rap", "spoken word"]
    if row["liveness"] > 0.7:
        words += ["live recording"]
    if row["loudness"] > -5:
        words += ["loud"]
    genres = " / ".join(g.replace("-", " ") for g in row["genres"])
    # Artist names are left out on purpose: "Be More Chill Ensemble" shouldn't match "chill".
    return f"{genres} song. {', '.join(words).capitalize()}."


def audio_matrix(df: pd.DataFrame, genre_weight: float = 0.6) -> np.ndarray:
    X = StandardScaler().fit_transform(df[config.AUDIO_FEATURES].to_numpy(dtype=float))
    X = X / np.sqrt(len(config.AUDIO_FEATURES))  # audio block has unit-ish length
    if genre_weight > 0:
        G = MultiLabelBinarizer().fit_transform(df["genres"]).astype(float)
        G = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)
        X = np.hstack([X, genre_weight * G])
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str], show_progress: bool = False) -> np.ndarray:
    return embedder().encode(texts, normalize_embeddings=True, batch_size=128,
                             show_progress_bar=show_progress).astype(np.float32)


def text_matrix(df: pd.DataFrame, show_progress: bool = False) -> np.ndarray:
    """Cached to data/text_embeddings.npy - building takes a few minutes, once."""
    if EMB_PATH.exists():
        E = np.load(EMB_PATH).astype(np.float32)
        if len(E) == len(df):
            return E
    E = embed(df["description"].tolist(), show_progress)
    np.save(EMB_PATH, E.astype(np.float16))  # half precision: half the disk, same rankings
    return E
