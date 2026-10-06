"""Fetch and clean the UCI Auto MPG dataset (id=9)."""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLEAN_CSV = DATA_DIR / "autompg_clean.csv"


def load_autompg_clean() -> pd.DataFrame:
    if CLEAN_CSV.exists():
        return pd.read_csv(CLEAN_CSV)

    autompg = fetch_ucirepo(id=9)
    df = pd.concat([autompg.data.features, autompg.data.targets], axis=1)
    df = df.dropna(subset=["horsepower"]).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    return df


if __name__ == "__main__":
    df = load_autompg_clean()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(df.head())
