"""Load and sample the ToxicChat dataset (real user prompts sent to a chatbot, human-labelled)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET = "lmsys/toxic-chat"
CONFIG = "toxicchat0124"


def stratified_sample(df: pd.DataFrame, n: int | None, label_col: str, seed: int) -> pd.DataFrame:
    """Return up to n rows, keeping the class ratio of label_col."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    sample, _ = train_test_split(df, train_size=n, stratify=df[label_col], random_state=seed)
    return sample.reset_index(drop=True)


def clean(df: pd.DataFrame, text_col: str = "user_input") -> pd.DataFrame:
    """Drop empty prompts and exact duplicates."""
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].str.strip().str.len() > 0]
    return df.drop_duplicates(subset=[text_col]).reset_index(drop=True)


def load_toxicchat(n_train: int | None = 2000, n_test: int | None = 1000, seed: int = 0):
    """Return (train_df, test_df) with columns: text, label (1 = toxic), jailbreaking."""
    from datasets import load_dataset  # imported here so tests run without `datasets`

    ds = load_dataset(DATASET, CONFIG)
    out = []
    for split, n in (("train", n_train), ("test", n_test)):
        df = ds[split].to_pandas()[["user_input", "toxicity", "jailbreaking"]]
        df = clean(df).rename(columns={"user_input": "text", "toxicity": "label"})
        out.append(stratified_sample(df, n, "label", seed))
    return out[0], out[1]
