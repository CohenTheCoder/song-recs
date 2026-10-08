"""Streamlit front end.  Run:  streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import tour
from song_recs import Recommender, Request
from song_recs.features import EMB_PATH

ROOT = Path(__file__).parent
RADAR = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness"]

st.set_page_config(page_title="Song Recs", page_icon="🎧", layout="wide")
ss = st.session_state
ss.setdefault("liked", [])
ss.setdefault("skipped", [])


@st.cache_resource(show_spinner=False)
def get_rec(genre_weight: float) -> Recommender:
    return Recommender(genre_weight=genre_weight)


@st.cache_data(show_spinner=False)
def catalog_map(genre_weight: float, n: int = 4000) -> pd.DataFrame:
    """2-D picture of the audio space: PCA squashes each song's vector to x, y."""
    from sklearn.decomposition import PCA

    rec = get_rec(genre_weight)
    idx = np.random.default_rng(0).choice(len(rec.df), min(n, len(rec.df)), replace=False)
    xy = PCA(n_components=2, random_state=0).fit(rec.A[idx]).transform(rec.A)
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}), idx


def esc(text) -> str:
    """Streamlit renders text between two $ signs as LaTeX math - escape them."""
    return str(text).replace("$", "\\$")


def spotify(track_id: str) -> str:
    return f"https://open.spotify.com/track/{track_id}"


# ---------- sidebar ----------
st.sidebar.title("🎧 Song Recs")
st.sidebar.caption("Content-based + natural-language music recommendations")
page = st.sidebar.radio("Steps", ["👋 Start here", "1 · Recommend", "2 · Map of the catalog", "How it works"],
                        key="page")
st.sidebar.divider()
st.sidebar.markdown("**Tuning knobs**")
text_weight = st.sidebar.slider("Sound ⟷ description", 0.0, 1.0, 0.4,
                                help="0 = match audio features only · 1 = match the text-embedding of each song's description")
diversity = st.sidebar.slider("Variety (MMR)", 0.0, 1.0, 0.2, help="Penalize songs too similar to ones already picked")
genre_weight = st.sidebar.slider("Stick to genre", 0.0, 1.5, 0.6, 0.1, help="Weight of genre tags in the audio vector")
pop_range = st.sidebar.slider("Popularity range", 0, 100, (0, 100), help="Lower the max to find deep cuts")
max_artist = st.sidebar.number_input("Max songs per artist", 1, 10, 2)
k = st.sidebar.number_input("How many", 5, 30, 10)

if not EMB_PATH.exists():
    st.warning("First run: building the song index (downloads 114k tracks from Hugging Face and embeds "
               "30k descriptions, ~4 min once). Or run `python scripts/build_index.py`.")
with st.spinner("Loading catalog…"):
    rec = get_rec(genre_weight)
df = rec.df


def add(i: int, where: str) -> None:
    other = "skipped" if where == "liked" else "liked"
    if i in ss[other]:
        ss[other].remove(i)
    if i not in ss[where]:
        ss[where].append(i)


# ======================= START HERE =======================
def _start_demo():
    for q in ("Hotel California", "Free Fallin'"):
        hits = rec.search(q, 1)
        if len(hits):
            add(int(hits.index[0]), "liked")
    ss.page = "1 · Recommend"


if page.startswith("👋"):
    tour.render(
        kicker="Recommender systems · Embeddings · NLP",
        title="🎧 Song Recs",
        subtitle="Tell it songs you love, or just describe a vibe, and get music that fits, "
                 "with a reason for every pick.",
        gradient=("#db2777", "#7c3aed"), accent="#db2777",
        steps=[
            {"emoji": "💿", "title": "Load 30,000 songs",
             "text": "Real Spotify tracks with measurements like energy, danceability, mood and tempo.",
             "hood": "Hugging Face dataset, de-duplicated"},
            {"emoji": "🎚️", "title": "Turn sound into numbers",
             "text": "Each song becomes a point in “sound space”. Songs that sound alike sit close together.",
             "hood": "standardized audio-feature vectors"},
            {"emoji": "✍️", "title": "Describe every song in words",
             "text": "“Rock song. Slow, sad, acoustic.” Now you can search the catalog by typing a vibe.",
             "hood": "🤗 sentence-transformer embeddings"},
            {"emoji": "❤️", "title": "Learn your taste",
             "text": "Your taste = the average of songs you like, nudged away from songs you skip.",
             "hood": "Rocchio relevance feedback"},
            {"emoji": "🌈", "title": "Keep it varied",
             "text": "No ten copies of the same song. Variety is balanced against relevance, max 2 per artist.",
             "hood": "Maximal Marginal Relevance (MMR)"},
            {"emoji": "💬", "title": "Explain every pick",
             "text": "“Similar energy and mood; genre: folk.” Recommendations you can trust.",
             "hood": "feature-level explanations"},
        ],
        chips=["🤗 all-MiniLM-L6-v2", "🤗 Spotify tracks dataset", "scikit-learn", "NumPy", "Streamlit", "Plotly"],
    )
    st.button("▶  Try it: songs like Hotel California + Free Fallin'", type="primary", on_click=_start_demo)
    st.caption("Or head to step 1 and search any song, or type a vibe like “rainy day jazz”.")

# ======================= RECOMMEND =======================
elif page.startswith("1"):
    st.header("Tell me what you like")
    c1, c2 = st.columns(2)
    with c1:
        q = st.text_input("🔎 Search a song or artist", placeholder="e.g. Hotel California, Taylor Swift, Daft Punk")
        if q:
            hits = rec.search(q, 8)
            if hits.empty:
                st.caption("No match in the 30k-song catalog.")
            for i, r in hits.iterrows():
                a, b = st.columns([5, 1])
                a.markdown(f"**{esc(r['track_name'])}** · {esc(r['artists'])}  \n<small>{r['genre']} · popularity {r['popularity']}</small>",
                           unsafe_allow_html=True)
                if b.button("➕", key=f"add-{i}", help="Add to songs you like"):
                    add(int(i), "liked")
                    st.rerun()
    with c2:
        vibe = st.text_input("💬 …and/or describe a vibe", placeholder="chill acoustic for a rainy afternoon")
        st.caption("Your words are embedded by a Hugging Face sentence-transformer and matched against a "
                   "written description of every song.")
        if ss.liked:
            st.markdown("**Songs you like**")
            for i in list(ss.liked):
                a, b = st.columns([5, 1])
                a.markdown(esc(df.at[i, "label"]))
                if b.button("✕", key=f"rm-{i}"):
                    ss.liked.remove(i)
                    st.rerun()
        if ss.skipped:
            st.caption(f"Skipped: {len(ss.skipped)} songs (used to steer away)")
        if (ss.liked or ss.skipped) and st.button("Reset"):
            ss.liked, ss.skipped = [], []
            st.rerun()

    req = Request(liked=list(ss.liked), skipped=list(ss.skipped), vibe=vibe.strip(), k=int(k),
                  text_weight=text_weight, diversity=diversity, max_per_artist=int(max_artist),
                  popularity=pop_range)
    st.divider()
    if not req.liked and not req.vibe:
        st.subheader("Popular right now")
        st.caption("Add a song you like or describe a vibe to personalize.")
    else:
        st.subheader("Recommended for you")
        st.caption("👍 = more like this · 👎 = less like this. Each click updates your taste vector instantly.")
    with st.spinner("Scoring 30,000 songs…"):
        out = rec.recommend(req)

    left, right = st.columns([3, 2])
    with left:
        for i, r in out.iterrows():
            a, b, c = st.columns([8, 1, 1])
            a.markdown(f"**[{esc(r['track_name'])}]({spotify(r['track_id'])})** · {esc(r['artists'])}  \n"
                       f"<small>{r['genre']} · popularity {r['popularity']} · match {r['match']:.2f} · "
                       f"<i>{r['why']}</i></small>", unsafe_allow_html=True)
            if b.button("👍", key=f"up-{i}"):
                add(int(i), "liked")
                st.rerun()
            if c.button("👎", key=f"down-{i}"):
                add(int(i), "skipped")
                st.rerun()
    with right:
        fig = go.Figure()
        if req.liked:
            prof = rec.taste_profile(req.liked)
            fig.add_trace(go.Scatterpolar(r=prof[RADAR].tolist(), theta=RADAR, fill="toself", name="Your likes"))
        recp = out[RADAR].mean()
        fig.add_trace(go.Scatterpolar(r=recp.tolist(), theta=RADAR, fill="toself", name="Recommendations"))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)), title="Sound profile",
                          height=420, margin=dict(t=60, b=30, l=70, r=70),
                          legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(out[["track_name", "artists", "tempo", "energy", "valence"]].round(2),
                     hide_index=True, use_container_width=True)


# ======================= MAP =======================
elif page.startswith("2"):
    st.header("Map of the catalog")
    st.caption("Each dot is a song. PCA squashes the ~120-number audio vector down to 2 numbers, so songs "
               "that sound alike land near each other. Showing a 4,000-song sample plus your likes.")
    xy, idx = catalog_map(genre_weight)
    top_genres = df["genre"].value_counts().head(12).index
    sample = df.iloc[idx].assign(x=xy["x"].iloc[idx].values, y=xy["y"].iloc[idx].values)
    sample["genre_shown"] = sample["genre"].where(sample["genre"].isin(top_genres), "other")
    sample = sample.sort_values("genre_shown", key=lambda g: g != "other")  # draw "other" underneath
    fig = px.scatter(sample, x="x", y="y", color="genre_shown", hover_name="label", opacity=0.6,
                     hover_data=["energy", "valence", "acousticness"], height=620,
                     color_discrete_map={"other": "#e2e8f0"})
    fig.update_traces(marker=dict(size=5))
    if ss.liked:
        liked = df.iloc[ss.liked].assign(x=xy["x"].iloc[ss.liked].values, y=xy["y"].iloc[ss.liked].values)
        fig.add_trace(go.Scatter(x=liked["x"], y=liked["y"], mode="markers+text", text=liked["track_name"],
                                 textposition="top center", name="your likes",
                                 marker=dict(size=14, symbol="star", color="#0f172a")))
    st.plotly_chart(fig, use_container_width=True)


# ======================= HOW IT WORKS =======================
else:
    st.header("How it works")
    st.markdown((ROOT / "docs/PROCESS.md").read_text())
    res = ROOT / "docs/results.md"
    if res.exists():
        st.markdown(res.read_text())
