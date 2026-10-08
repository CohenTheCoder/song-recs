# Offline evaluation (300 random seed songs, top-10)

| model                 |   genre_precision@10 |   diversity |   unique_recs |   avg_popularity |
|:----------------------|---------------------:|------------:|--------------:|-----------------:|
| random                |                0.027 |       0.989 |         0.956 |           55.619 |
| most popular          |                0.032 |       0.707 |         0.003 |           97.2   |
| audio kNN             |                0.159 |       0.063 |         0.952 |           55.433 |
| audio kNN + MMR (0.2) |                0.158 |       0.064 |         0.951 |           55.446 |
| audio kNN + MMR (0.5) |                0.16  |       0.068 |         0.954 |           55.368 |

Catalog average popularity: 55.5. Audio models are built without genre tags, so genre precision measures whether sound alone finds same-genre music.
