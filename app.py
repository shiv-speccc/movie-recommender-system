"""
app.py
======
Streamlit web app for the Movie Recommender System.

Run:
    streamlit run app.py

Prerequisites:
    - Train models first: python src/train.py
    - Or provide the MovieLens data in ./data/
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_loader import load_and_merge, rating_matrix, popularity_stats
from recommender import CollaborativeFilteringRecommender, SVDRecommender, HybridRecommender
import eda as eda_module

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #e50914; text-align: center; margin-bottom: 0.2rem; }
    .sub-header  { font-size: 1rem;   color: #aaa;       text-align: center; margin-bottom: 1.5rem; }
    .metric-card { background: #1a1a2e; border-radius: 12px; padding: 1rem; text-align: center; }
    .rec-card    { background: #16213e; border-left: 4px solid #e50914; border-radius: 8px;
                   padding: 0.8rem 1rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data / model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    df     = load_and_merge()
    stats  = popularity_stats(df)
    matrix = rating_matrix(df)
    return df, stats, matrix


@st.cache_resource(show_spinner="Training models …")
def get_models(matrix, stats):
    cf      = CollaborativeFilteringRecommender(min_ratings=50).fit(matrix, stats)
    svd     = SVDRecommender(n_components=50, min_ratings=50).fit(matrix, stats)
    hybrid  = HybridRecommender(cf_weight=0.5, svd_weight=0.5, min_ratings=50).fit(matrix, stats)
    return cf, svd, hybrid


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🎬 Movie Recommender System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">MovieLens 100K · Collaborative Filtering · SVD · Hybrid Ensemble</p>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
try:
    df, stats, matrix = load_data()
    cf_model, svd_model, hybrid_model = get_models(matrix, stats)
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.error(
        "**Data not found.**  "
        "Please download the MovieLens 100K dataset from Kaggle and place the files in `./data/`.\n\n"
        "Required files: `u.data`, `u.item`, `u.user`\n\n"
        "Dataset: https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset"
    )

if data_loaded:
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        model_choice = st.radio(
            "Recommendation Model",
            ["Hybrid (CF + SVD)", "Collaborative Filtering", "SVD (Matrix Factorisation)"],
        )
        top_n = st.slider("Number of recommendations", 5, 20, 10)
        st.markdown("---")
        st.markdown("**About**")
        st.caption(
            "This app uses the MovieLens 100K dataset to demonstrate three "
            "collaborative filtering strategies: memory-based (Pearson CF), "
            "model-based (TruncatedSVD), and a hybrid ensemble."
        )

    MODEL_MAP = {
        "Collaborative Filtering":    cf_model,
        "SVD (Matrix Factorisation)": svd_model,
        "Hybrid (CF + SVD)":          hybrid_model,
    }
    active_model = MODEL_MAP[model_choice]

    # ── Top metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Ratings",  f"{len(df):,}")
    c2.metric("Unique Users",   f"{df['user_id'].nunique():,}")
    c3.metric("Unique Movies",  f"{df['item_id'].nunique():,}")
    sparsity = 1 - matrix.notna().mean().mean()
    c4.metric("Matrix Sparsity", f"{sparsity:.1%}")

    st.divider()

    # ── Recommender UI ────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔍 Get Recommendations", "📊 Explore Data", "🏆 Top Movies"])

    with tab1:
        all_movies = sorted(matrix.columns.tolist())
        selected   = st.selectbox("Choose a movie you like:", all_movies,
                                  index=all_movies.index("Star Wars (1977)") if "Star Wars (1977)" in all_movies else 0)

        if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner(f"Finding movies similar to *{selected}* …"):
                try:
                    recs = active_model.recommend(selected, top_n=top_n)
                    st.subheader(f"Movies similar to **{selected}**")
                    st.caption(f"Model: {model_choice}")

                    score_col = [c for c in recs.columns if "score" in c or "corr" in c][0]

                    for _, row in recs.iterrows():
                        score_val = row[score_col]
                        st.markdown(
                            f'<div class="rec-card">'
                            f'<b>#{int(row.name)}</b> &nbsp; {row["title"]} &nbsp; '
                            f'<span style="color:#aaa; font-size:0.85rem;">'
                            f'Score: {score_val:.3f} &nbsp;|&nbsp; {int(row["rating_count"]):,} ratings</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                except ValueError as e:
                    st.error(str(e))

    with tab2:
        plot_choice = st.selectbox("Select EDA plot:", [
            "Rating Distribution",
            "Rating vs. Popularity",
            "Genre Distribution",
            "User Activity",
            "Matrix Sparsity Heatmap",
            "SVD Variance",
        ])
        fig = None
        if   plot_choice == "Rating Distribution":       fig = eda_module.plot_rating_distribution(df)
        elif plot_choice == "Rating vs. Popularity":     fig = eda_module.plot_rating_vs_popularity(stats)
        elif plot_choice == "Genre Distribution":        fig = eda_module.plot_genre_distribution(df)
        elif plot_choice == "User Activity":             fig = eda_module.plot_user_activity(df)
        elif plot_choice == "Matrix Sparsity Heatmap":  fig = eda_module.plot_sparsity_heatmap(matrix)
        elif plot_choice == "SVD Variance":              fig = eda_module.plot_svd_variance(svd_model)
        if fig:
            st.pyplot(fig, use_container_width=True)

    with tab3:
        n_top = st.slider("Show top N movies", 10, 50, 20)
        top_df = stats.nlargest(n_top, "rating_count")[["title", "rating_count", "mean_rating", "bayesian_avg"]]
        top_df.columns = ["Title", "# Ratings", "Mean Rating", "Bayesian Avg"]
        top_df = top_df.reset_index(drop=True)
        top_df.index += 1
        st.dataframe(top_df, use_container_width=True, height=600)
