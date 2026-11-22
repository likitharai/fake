"""Dataset loading, EDA, cleaning, sampling and stratified splitting."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import LABEL_MAP, RAW_DATA_PATH
from src.preprocessing import build_input_text

REQUIRED_COLUMNS = {"title", "text", "label"}


def load_raw_dataset(path=RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download WELFake_Dataset.csv and place it there."
        )
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]  # drop CSV index column
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def run_eda(df: pd.DataFrame) -> dict:
    """Exploratory statistics on the raw data (computed before any cleaning)."""
    words = df["text"].fillna("").astype(str).str.split().str.len()
    counts = df["label"].value_counts(dropna=True).to_dict()
    return {
        "raw_rows": int(len(df)),
        "missing_title": int(df["title"].isna().sum()),
        "missing_text": int(df["text"].isna().sum()),
        "missing_label": int(df["label"].isna().sum()),
        "duplicate_title_text": int(df.duplicated(subset=["title", "text"]).sum()),
        "label_counts_raw": {
            LABEL_MAP[0]: int(counts.get(0, 0)),
            LABEL_MAP[1]: int(counts.get(1, 0)),
        },
        "avg_words_raw": round(float(words.mean()), 1),
        "median_words_raw": float(words.median()),
    }


def clean_dataframe(df: pd.DataFrame):
    """Clean text, drop empties, drop duplicates and conflicting labels.

    Deduplication happens BEFORE splitting so identical articles cannot
    leak between train and test.
    Returns (clean_df, number_of_duplicates_removed).
    """
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin(LABEL_MAP.keys())]
    df["input_text"] = [build_input_text(t, x) for t, x in zip(df["title"], df["text"])]
    df = df[df["input_text"].str.len() > 0]

    # Same text with contradictory labels is unreliable: drop every copy.
    conflicts = df.groupby("input_text")["label"].nunique()
    df = df[~df["input_text"].isin(conflicts[conflicts > 1].index)]

    before = len(df)
    df = df.drop_duplicates(subset=["input_text"])
    removed = before - len(df)
    return df[["input_text", "label"]].reset_index(drop=True), int(removed)


def maybe_subsample(df: pd.DataFrame, n, seed: int) -> pd.DataFrame:
    """Stratified subsample for quick runs (no-op when n is None or too large)."""
    if not n or n >= len(df):
        return df
    sub, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sub.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, seed: int, test_size: float, val_size: float):
    """Return (train, val, test) with class proportions preserved."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    val_relative = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relative, stratify=train_val["label"], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
