"""
predict.py

Load saved model and vectorizer, accept a text input from the user, and print whether the article is 'Fake' or 'Real'.

Run:
python src/predict.py
"""
import sys
from pathlib import Path
import joblib

# Make sure local src modules can be imported when running from the project root
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import clean_text


def main():
    models_dir = Path("models")
    pipeline_path = models_dir / "news_pipeline.pkl"
    model_path = models_dir / "news_model.pkl"
    vec_path = models_dir / "tfidf_vectorizer.pkl"

    # Prefer loading the pipeline if available
    if pipeline_path.exists():
        pipeline = joblib.load(pipeline_path)
        use_pipeline = True
    else:
        if not model_path.exists() or not vec_path.exists():
            raise FileNotFoundError("Model or vectorizer not found in the 'models' folder. Run 'python src/train.py' first.")
        clf = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        use_pipeline = False

    print("Enter/paste the news article text. Press Enter twice to submit (or Ctrl+C to exit).")
    # Collect multi-line input if user wants (stop on empty line)
    lines = []
    try:
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
    except KeyboardInterrupt:
        print("\nAborted by user")
        return

    text = "\n".join(lines).strip()
    if not text:
        print("No text provided. Exiting.")
        return

    text_clean = clean_text(text)
    if use_pipeline:
        pred = pipeline.predict([text_clean])[0]
    else:
        X_vec = vectorizer.transform([text_clean])
        pred = clf.predict(X_vec)[0]

    # Normalize output to capitalized form
    if isinstance(pred, str):
        label = pred.capitalize()
    else:
        label = str(pred)

    print(f"Prediction: {label}")


if __name__ == "__main__":
    main()
