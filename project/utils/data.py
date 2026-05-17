import ast
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DriftConfig


def load_reviews(path: str | Path, date_col: str = "review_date") -> pd.DataFrame:
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    df["year"] = df[date_col].dt.year

    return df


def parse_embedding_column(series: pd.Series) -> np.ndarray:
    def parse_one(value):
        if isinstance(value, str):
            return ast.literal_eval(value)
        return value

    return np.vstack(series.apply(parse_one).values).astype(np.float32)


def build_time_masks(
    df: pd.DataFrame,
    config: DriftConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = df["year"] <= config.train_end
    gap_mask = (df["year"] > config.train_end) & (df["year"] <= config.gap_end)
    test_mask = df["year"] > config.gap_end

    return train_mask.values, gap_mask.values, test_mask.values


def scale_embeddings(X: np.ndarray, train_mask: np.ndarray):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X[train_mask])
    X_scaled = scaler.transform(X)

    return X_train_scaled.astype(np.float32), X_scaled.astype(np.float32), scaler
