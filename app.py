"""
Flask Web Application for News Classification (Fake vs Real)

Run:
python app.py
Then open http://localhost:5000 in your browser
"""
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import joblib
import sys

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from src.preprocess import clean_text

app = Flask(__name__)

# Load the model at startup
models_dir = Path("models")
pipeline_path = models_dir / "news_pipeline.pkl"

if not pipeline_path.exists():
    raise FileNotFoundError("Model not found. Please run 'python src/train.py' first.")

pipeline = joblib.load(pipeline_path)


@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please enter some text'}), 400
        
        # Clean and predict
        text_clean = clean_text(text)
        prediction = pipeline.predict([text_clean])[0]
        probabilities = pipeline.predict_proba([text_clean])[0]
        
        # Get probability scores
        classes = pipeline.classes_
        proba_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        
        # Normalize prediction
        label = prediction.capitalize() if isinstance(prediction, str) else str(prediction)
        
        confidence = max(probabilities) * 100
        
        return jsonify({
            'prediction': label,
            'confidence': round(confidence, 2),
            'probabilities': proba_dict
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 News Classifier Web App Starting...")
    print("="*60)
    print("📍 Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
