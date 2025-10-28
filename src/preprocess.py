"""
preprocess.py

Text cleaning utilities using nltk.
- lowercasing
- remove punctuation and numbers
- remove stopwords

Functions:
- clean_text(text) -> str
- preprocess_series(series) -> pd.Series

This module will attempt to download nltk resources (punkt, stopwords) if missing.
"""
import re
from typing import Optional

import nltk
from nltk.corpus import stopwords
import pandas as pd


def _ensure_nltk_resources():
    """Download required NLTK resources if they are missing."""
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


# Initialize resources on import
_ensure_nltk_resources()

EN_STOPWORDS = set(stopwords.words("english"))

# Regex to remove URLs, punctuation and numbers. We'll keep whitespace and letters.
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_ALPHANUM_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    """Clean a single text string.

    Steps:
    - Handle None -> ''
    - Lowercase
    - Remove URLs
    - Remove non-alphabetic characters
    - Tokenize and remove stopwords
    - Return cleaned string
    """
    if text is None:
        return ""
    # Ensure it's a string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = URL_RE.sub(" ", text)

    # Remove punctuation and numbers
    text = NON_ALPHANUM_RE.sub(" ", text)

    # Collapse whitespace
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    # Simple tokenization using regex to avoid heavy NLTK tokenizer dependencies
    # Extract alphabetic tokens and remove stopwords
    tokens = re.findall(r"\b[a-zA-Z]+\b", text)
    tokens = [t for t in tokens if t not in EN_STOPWORDS]

    return " ".join(tokens)


def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply clean_text to a pandas Series of texts.

    Returns a new pd.Series of cleaned strings.
    """
    return series.fillna("").astype(str).apply(clean_text)
