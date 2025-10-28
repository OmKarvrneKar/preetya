"""
data_loader.py

Load dataset and split into train/test sets (80/20).

Functions:
- load_dataframe(path): returns a pandas DataFrame
- load_and_split(path, test_size, random_state): returns X_train, X_test, y_train, y_test

Assumes CSV has columns 'text' and 'label'.
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataframe(path: str = "data/news.csv") -> pd.DataFrame:
    """Load the CSV into a pandas DataFrame and keep only required columns.

    Args:
        path: path to CSV file (default: data/news.csv)

    Returns:
        pd.DataFrame with columns ['text', 'label']
    """
    df = pd.read_csv(path)

    # Keep only the columns we need and drop rows with missing values in those columns
    expected_cols = ["text", "label"]
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' not found in {path}")

    df = df[expected_cols].copy()
    df = df.dropna(subset=expected_cols)
    df = df.reset_index(drop=True)
    return df


def load_and_split(path: str = "data/news.csv", test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load the dataset and return stratified train/test splits.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df = load_dataframe(path)
    X = df["text"]
    y = df["label"]

    # Use stratified split to keep class balance in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Reset indices for convenience
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test
