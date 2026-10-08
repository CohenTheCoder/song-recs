### The pipeline in 5 steps

```
Hugging Face dataset (114k Spotify tracks)
        │  1. download + merge duplicate versions + keep 30k most popular
        ▼
   ┌────────────────────────────┬──────────────────────────────────────────────┐
   │ 2a. AUDIO vector           │ 2b. TEXT vector                              │
   │ 9 audio features (z-score) │ "rock song. Slow, sad, acoustic."            │
   │ + genre tags × weight      │  → all-MiniLM-L6-v2 → 384 numbers            │
   └─────────────┬──────────────┴───────────────────────┬──────────────────────┘
                 │   3. taste = avg(likes) − ½·avg(skips) │  ← or your typed "vibe"
                 ▼                                        ▼
        cosine similarity to all 30k songs, blended by the "sound ⟷ description" slider
                 │
                 ▼  4. shortlist 400 → MMR re-rank for variety → max N per artist
        top-k recommendations + a reason for each
                 │
                 ▼  5. 👍 / 👎 updates the taste vector instantly (feedback loop)
```

**Step 1 · Data** (`song_recs/data.py`)
`hf_hub_download("maharshipandya/spotify-tracks-dataset", "dataset.csv")` pulls a 20 MB CSV of
Spotify tracks with audio features. The same song shows up many times (once per genre tag, plus
"Remastered 2011" versions), so we normalize titles and merge duplicates while keeping every genre.

**Step 2a · Audio vectors** (`song_recs/features.py`)
Danceability, energy, valence (musical positivity), acousticness and the rest get
**standardized** (z-scores), because otherwise tempo (60–200) would dominate valence (0–1). Genre
tags are one-hot encoded and multiplied by the "stick to genre" weight. Rows are scaled to unit
length, so a dot product equals **cosine similarity**.

**Step 2b · Text vectors: the NLP part**
Each song gets a generated description ("hip hop song. Fast, high-energy, danceable, rap.").
A **sentence-transformer from Hugging Face** turns it into a 384-dimensional embedding. Your
typed vibe ("songs for a late-night drive") is embedded the same way, so text and songs can be
compared directly. This is the same idea behind semantic search.

**Step 3 · Taste vector**
Your taste = average vector of liked songs − ½ × average of skipped songs. This is the classic
**Rocchio** relevance-feedback update from information retrieval. Every song is scored by cosine
similarity to your taste in both spaces, blended by the slider, plus a small popularity nudge.

**Step 4 · Re-rank for variety: Maximal Marginal Relevance (MMR)**
The top 10 by pure similarity tend to sound almost identical. MMR picks songs one at a time,
maximizing `λ × relevance − (1 − λ) × similarity to songs already picked`. A cap of N songs per
artist stops one artist from taking over the list.

**Step 5 · Feedback loop**
👍 adds a song to your likes and 👎 adds it to your skips. The taste vector moves and the list
updates. With real users you'd log these events and train a collaborative-filtering model
(e.g., matrix factorization) on top: "people who liked X also liked Y".

**Evaluation** (`scripts/evaluate.py`)
With no real listening logs, we use a proxy: *given one seed song, how many of the top-10 share
its genre?* For fairness, the audio model is built **without** genre tags, so it must find
same-genre songs from sound alone. We also measure **diversity**, **catalog coverage** (does it
only recommend the same few songs?) and **popularity bias**, and compare against random and
most-popular baselines.
