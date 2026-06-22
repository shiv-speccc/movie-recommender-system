"""
eda.py
======
Exploratory Data Analysis functions that produce matplotlib figures.
All functions return the figure object so callers can save or show.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams['axes.formatter.use_mathtext'] = False
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (12, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Rating distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_rating_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    # Count per rating value
    rating_counts = df["rating"].value_counts().sort_index()
    axes[0].bar(rating_counts.index, rating_counts.values, color="steelblue", edgecolor="white")
    axes[0].set_title("Rating Value Distribution")
    axes[0].set_xlabel("Rating (1–5)")
    axes[0].set_ylabel("Count")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Distribution of number-of-ratings per movie
    stats = df.groupby("title")["rating"].count()
    axes[1].hist(stats, bins=60, color="coral", edgecolor="white")
    axes[1].set_title("Ratings per Movie (distribution)")
    axes[1].set_xlabel("Number of Ratings")
    axes[1].set_ylabel("Number of Movies")
    axes[1].set_yscale("log")

    fig.suptitle("Rating Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Top-N most rated movies
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_movies(stats: pd.DataFrame, n: int = 20) -> plt.Figure:
    top = stats.nlargest(n, "rating_count")
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(top["title"][::-1], top["rating_count"][::-1], color="teal", edgecolor="white")
    ax.set_xlabel("Number of Ratings")
    ax.set_title(f"Top {n} Most-Rated Movies", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar in bars:
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width()):,}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Mean rating vs number of ratings scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_rating_vs_popularity(stats: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sc = ax.scatter(
        stats["rating_count"], stats["mean_rating"],
        alpha=0.4, c=stats["bayesian_avg"], cmap="plasma", edgecolors="none", s=30
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Bayesian Avg Rating")
    ax.set_xlabel("Number of Ratings (popularity)")
    ax.set_ylabel("Mean Rating")
    ax.set_title("Popularity vs. Mean Rating\n(colour = Bayesian average)", fontweight="bold")
    ax.set_xscale("log")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Genre distribution
# ─────────────────────────────────────────────────────────────────────────────

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def plot_genre_distribution(df: pd.DataFrame) -> plt.Figure:
    available = [g for g in GENRE_COLS if g in df.columns]
    if not available:
        raise ValueError("Genre columns not found. Make sure to use the merged DataFrame.")

    genre_counts = df[available].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    genre_counts.plot(kind="barh", ax=ax, color="mediumpurple", edgecolor="white")
    ax.set_title("Movies per Genre", fontweight="bold")
    ax.set_xlabel("Number of Movies (multi-genre counted in each)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# User activity distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_user_activity(df: pd.DataFrame) -> plt.Figure:
    user_counts = df.groupby("user_id")["rating"].count()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(user_counts, bins=60, color="darkorange", edgecolor="white")
    ax.set_title("Ratings per User (distribution)", fontweight="bold")
    ax.set_xlabel("Number of Ratings per User")
    ax.set_ylabel("Number of Users")
    ax.set_yscale("log")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sparsity heatmap (sample)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sparsity_heatmap(matrix: pd.DataFrame, n_users: int = 50, n_movies: int = 50) -> plt.Figure:
    """Visualise a sub-sample of the user-item matrix to show sparsity."""
    top_movies = matrix.notna().sum().nlargest(n_movies).index
    sub = matrix.loc[:, top_movies].iloc[:n_users]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        sub.notna().astype(int), ax=ax,
        cmap="Blues", cbar=False, linewidths=0, xticklabels=False, yticklabels=False
    )
    ax.set_title(
        f"Rating Matrix Sparsity (sample: {n_users} users × {n_movies} movies)\n"
        f"Blue = rated  |  White = missing  |  "
        f"Overall density: {matrix.notna().mean().mean():.2%}",
        fontweight="bold"
    )
    ax.set_xlabel("Movies (top rated)")
    ax.set_ylabel("Users")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SVD explained variance
# ─────────────────────────────────────────────────────────────────────────────

def plot_svd_variance(svd_model) -> plt.Figure:
    """Plot cumulative explained variance for the fitted SVD model."""
    ratios = svd_model._svd.explained_variance_ratio_
    cumsum = np.cumsum(ratios)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(cumsum) + 1), cumsum * 100, marker="o", markersize=3, color="steelblue")
    ax.axhline(80, linestyle="--", color="red", label="80% threshold")
    ax.set_xlabel("Number of Latent Factors")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("SVD Explained Variance Curve", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig
