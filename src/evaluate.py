"""
evaluate.py

Load saved model and vectorizer, run predictions on the test set, and print:
- accuracy
- precision
- recall
- f1-score
- confusion matrix

Run:
python src/evaluate.py
"""
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Make sure local src modules can be imported when running from the project root
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_and_split
from preprocess import preprocess_series


def main():
    models_dir = Path("models")
    pipeline_path = models_dir / "news_pipeline.pkl"
    model_path = models_dir / "news_model.pkl"
    vec_path = models_dir / "tfidf_vectorizer.pkl"

    # Prefer loading the saved pipeline if available
    if pipeline_path.exists():
        print("Loading pipeline...")
        pipeline = joblib.load(pipeline_path)
        use_pipeline = True
    else:
        # Fallback to individual components
        if not model_path.exists() or not vec_path.exists():
            raise FileNotFoundError("Model or vectorizer not found in the 'models' folder. Run 'python src/train.py' first.")
        print("Loading model and vectorizer...")
        clf = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        use_pipeline = False

    print("Loading test data (same split as training)...")
    X_train, X_test, y_train, y_test = load_and_split(path="data/news.csv")

    print("Preprocessing test texts...")
    X_test_clean = preprocess_series(X_test)

    print("Predicting on test set...")
    if use_pipeline:
        y_pred = pipeline.predict(X_test_clean)
    else:
        X_test_vec = vectorizer.transform(X_test_clean)
        y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    # Save a confusion matrix heatmap to the models directory for visual inspection
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path = models_dir / "confusion_matrix.png"
    plt.savefig(out_path)
    print(f"Saved confusion matrix heatmap to: {out_path}")


if __name__ == "__main__":
    main()
