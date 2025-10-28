"""
train.py

Train a fake news classifier.

Steps:
- Load train/test splits from data_loader
- Preprocess text with preprocess.preprocess_series
- Vectorize using TfidfVectorizer
- Train LogisticRegression
- Save model and vectorizer to /models using joblib

Run:
python src/train.py
"""
import os
import sys
from pathlib import Path

# Ensure the src directory is on sys.path so local imports work when running
# the scripts from the project root (e.g. `python src/train.py`). This makes
# imports like `from data_loader import ...` resolve correctly.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from data_loader import load_and_split
from preprocess import preprocess_series


def main():
    # Paths
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data and splitting (80/20)...")
    X_train, X_test, y_train, y_test = load_and_split(path="data/news.csv")

    print("Preprocessing text...")
    X_train_clean = preprocess_series(X_train)
    X_test_clean = preprocess_series(X_test)

    print("Vectorizing text with TF-IDF and creating pipeline...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))

    # Use class_weight='balanced' to mitigate class imbalance
    clf = LogisticRegression(solver="liblinear", random_state=42, max_iter=1000, class_weight='balanced')

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])

    print("Fitting pipeline on training data...")
    pipeline.fit(X_train_clean, y_train)

    # Evaluate quickly on the hold-out test set
    X_test_vec = X_test_clean  # pipeline will transform internally
    y_pred_test = pipeline.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred_test)

    # Save artifacts: pipeline plus individual components for compatibility
    pipeline_path = models_dir / "news_pipeline.pkl"
    model_path = models_dir / "news_model.pkl"
    vec_path = models_dir / "tfidf_vectorizer.pkl"

    joblib.dump(pipeline, pipeline_path)
    # Save individual components as well
    joblib.dump(pipeline.named_steps['clf'], model_path)
    joblib.dump(pipeline.named_steps['tfidf'], vec_path)

    print(f"Saved pipeline to: {pipeline_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved vectorizer to: {vec_path}")
    print(f"Test accuracy (quick check): {acc:.4f}")


if __name__ == "__main__":
    main()
