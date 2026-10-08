"""Settings in one place."""

DATASET_REPO = "maharshipandya/spotify-tracks-dataset"

# Keep the N most popular unique tracks. Bigger = more deep cuts but slower embedding build
# (~4 min per 10k songs on an older laptop CPU, done once).
CATALOG_SIZE = 30_000

# Sentence-embedding model for "describe a vibe" search.
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

AUDIO_FEATURES = ["danceability", "energy", "valence", "acousticness", "instrumentalness",
                  "speechiness", "liveness", "loudness", "tempo"]
