"""
data_loader.py
==============
Loads and preprocesses the MovieLens 100K dataset.

Dataset required:
    - u.data   → ratings (user_id, item_id, rating, timestamp) [TSV]
    - u.item   → movie metadata (item_id, title, release_date, genres…)
    - u.user   → user demographics (user_id, age, gender, occupation, zip)

Download: https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def load_ratings(path: str | None = None) -> pd.DataFrame:
    """Load raw ratings from u.data (tab-separated)."""
    fp = path or DATA_DIR / "u.data"
    df = pd.read_csv(
        fp,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def load_movies(path: str | None = None) -> pd.DataFrame:
    """Load movie metadata from u.item (pipe-separated)."""
    fp = path or DATA_DIR / "u.item"
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLS
    df = pd.read_csv(fp, sep="|", names=cols, encoding="latin-1", usecols=lambda c: c != "video_release_date")
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    return df


def load_users(path: str | None = None) -> pd.DataFrame:
    """Load user demographics from u.user (pipe-separated)."""
    fp = path or DATA_DIR / "u.user"
    df = pd.read_csv(
        fp,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"],
        encoding="latin-1",
    )
    return df


def load_and_merge(ratings_path=None, movies_path=None, users_path=None) -> pd.DataFrame:
    """
    Load all sources, merge them, and return a single enriched DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        user_id, item_id, rating, timestamp,
        title, release_year, genre columns...,
        age, gender, occupation
    """
    ratings = load_ratings(ratings_path)
    movies  = load_movies(movies_path)
    users   = load_users(users_path)

    df = (
        ratings
        .merge(movies[["item_id", "title", "release_year"] + GENRE_COLS], on="item_id", how="left")
        .merge(users[["user_id", "age", "gender", "occupation"]], on="user_id", how="left")
    )
    return df


def rating_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a user × movie pivot table (user-item matrix).
    Cells are mean ratings; NaN where a user hasn't rated a movie.
    """
    return df.pivot_table(index="user_id", columns="title", values="rating")


def popularity_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-movie statistics: mean rating, rating count, Bayesian avg.
    Bayesian average shrinks raw means toward the global mean for movies
    with very few ratings — a standard industry correction.
    """
    global_mean = df["rating"].mean()
    C = df.groupby("title")["rating"].count().quantile(0.70)   # confidence threshold

    stats = (
        df.groupby("title")["rating"]
        .agg(mean_rating="mean", rating_count="count")
        .reset_index()
    )
    # Bayesian average: (C * global_mean + count * mean) / (C + count)
    stats["bayesian_avg"] = (
        (C * global_mean + stats["rating_count"] * stats["mean_rating"])
        / (C + stats["rating_count"])
    )
    return stats.sort_values("bayesian_avg", ascending=False)
