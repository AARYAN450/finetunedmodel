"""Single source of truth for the train/test split.

Both baseline.py and train.py import this so the comparison is on identical
rows. Do not change SEED or TEST_SIZE after results are recorded.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
LABELED_CSV = Path(__file__).parent / "labeled.csv"


def load_split(path=LABELED_CSV):
    """Return (train_df, test_df), stratified on label."""
    df = pd.read_csv(path, encoding="utf-8")
    df = df.dropna(subset=["sentence", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    tr, te = load_split()
    print(f"train {len(tr)} rows, {tr.label.sum()} positives")
    print(f"test  {len(te)} rows, {te.label.sum()} positives")
