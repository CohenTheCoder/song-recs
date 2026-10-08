"""Step 1: download the song catalog from Hugging Face and clean it.

Dataset: https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
114,000 Spotify tracks across 114 genres, each with Spotify's audio features:

  danceability, energy, valence (musical happiness), acousticness, instrumentalness,
  speechiness, liveness   -> all 0..1
  loudness (dB), tempo (BPM), popularity (0..100)

The same song often appears several times (once per genre it's tagged with), so we merge
duplicates and keep every genre tag.
"""
import re
from pathlib import Path

import pandas as pd

from . import config

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data/catalog.parquet"


_VERSION = re.compile(
    r"\s*(?:[-–(\[]\s*)(?:\d{4}\s+)?(?:remaster(?:ed)?|live|acoustic|radio edit|single version|"
    r"mono|stereo|demo|edit|version|remix|feat\.|with |from )[^)\]]*[)\]]?.*$", re.I)


def base_title(title: str) -> str:
    """'Bohemian Rhapsody - Remastered 2011' -> 'bohemian rhapsody' so versions count as one song."""
    return _VERSION.sub("", title).strip().lower() or title.lower()


def download_raw() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(config.DATASET_REPO, "dataset.csv", repo_type="dataset")
    return pd.read_csv(path)


def clean(raw: pd.DataFrame, size: int = config.CATALOG_SIZE) -> pd.DataFrame:
    df = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
    df = df.dropna(subset=["track_name", "artists"])
    df["artists"] = df["artists"].str.replace(";", ", ")
    # merge duplicates: same song + artists (any version) -> one row, all genre tags kept
    df["song_key"] = df["track_name"].map(base_title) + " | " + df["artists"].str.lower()
    genres = df.groupby("song_key")["track_genre"].agg(lambda g: sorted(set(g)))
    df = (df.sort_values("popularity", ascending=False)
            .drop_duplicates("song_key")
            .drop(columns="track_genre")
            .merge(genres.rename("genres"), on="song_key"))
    df = df[df["duration_ms"].between(60_000, 15 * 60_000)]  # drop jingles and hour-long mixes
    df = df.nlargest(size, "popularity").reset_index(drop=True)
    df["genre"] = df["genres"].str[0]
    df["label"] = df["track_name"] + " — " + df["artists"]
    return df


def load_catalog() -> pd.DataFrame:
    """Cached cleaned catalog (built on first call)."""
    if CATALOG_PATH.exists():
        return pd.read_parquet(CATALOG_PATH)
    df = clean(download_raw())
    CATALOG_PATH.parent.mkdir(exist_ok=True)
    df.to_parquet(CATALOG_PATH)
    return df
