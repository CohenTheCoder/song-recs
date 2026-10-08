"""One-time setup: download the dataset from Hugging Face, clean it, and embed every song.

    python scripts/build_index.py

Creates data/catalog.parquet and data/text_embeddings.npy (a few minutes on a laptop CPU).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from song_recs import build_all  # noqa: E402

t = time.time()
rec = build_all(show_progress=True)
print(f"Catalog: {len(rec.df):,} songs · {rec.df['genre'].nunique()} genres · "
      f"text vectors {rec.E.shape} · built in {time.time() - t:.0f}s")
print("Example description:", rec.df["description"].iloc[0])
