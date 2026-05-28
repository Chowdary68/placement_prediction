"""
app.py - Flask Web Application for Placement Prediction
"""

from flask import Flask, render_template, request, jsonify
import joblib
import json
import pandas as pd
import os

# ─────────────────────────────────────────────────────────────
# Flask App Initialization
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# Base Directory
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
# Load Model Files Safely
# ─────────────────────────────────────────────────────────────
try:
    model = joblib.load(
        os.path.join(BASE_DIR, 'models', 'best_model.pkl')
    )

    scaler = joblib.load(
        os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    )

    with open(os.path.join(BASE_DIR, 'models', 'features.json'), 'r') as f:
        FEATURES = json.load(f)

    with open(os.path.join(BASE_DIR, 'models', 'model_info.json'), 'r') as f:
        MODEL_INFO = json.load(f)

except Exception as e:
    print(f"Error loading model files: {e}")

    model = None
    scaler = None
    FEATURES = []
    MODEL_INFO = {}

# ─────────────────────────────────────────────────────────────
# Home Route
# ─────────────────────────────────────────────────────────────
@app.route('/')
def home():

    best_model = MODEL_INFO.get('best_model', 'Unknown')

    stats = MODEL_INFO.get(best_model, {})

    return render_template(
        'index.html',
        best_model=best_model,
        stats=stats
    )

# ─────────────────────────────────────────────────────────────
# Prediction Route
# ─────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Check model availability
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model files not loaded properly'
            }), 500

        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No input data received'
            }), 400

        # Extract features safely
        values = []

        for feature in FEATURES:

            value = data.get(feature, 0)

            try:
                value = float(value)
            except:
                value = 0

            values.append(value)

        # Convert into DataFrame
        input_df = pd.DataFrame(
            [values],
            columns=FEATURES
        )

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Prediction
        prediction = int(model.predict(scaled_input)[0])

        # Probability
        probability = 0

        if hasattr(model, "predict_proba"):
            probability = float(
                model.predict_proba(scaled_input)[0][1]
            ) * 100

        # ─────────────────────────────────────────────────────
        # Suggestions / Tips
        # ─────────────────────────────────────────────────────
        tips = []

        if float(data.get('CGPA', 0)) < 7:
            tips.append(
                "Improve your CGPA — aim for 7.5+"
            )

        if int(float(data.get('Internships', 0))) == 0:
            tips.append(
                "Complete at least 1-2 internships"
            )

        if int(float(data.get('Projects', 0))) < 2:
            tips.append(
                "Build more projects (aim for 3+)"
            )

        if float(data.get('AptitudeScore', 0)) < 60:
            tips.append(
                "Practice aptitude & reasoning tests"
            )

        if int(float(data.get('Backlogs', 0))) > 0:
            tips.append(
                "Clear all backlogs before applying"
            )

        if int(float(data.get('CommunicationSkills', 0))) < 3:
            tips.append(
                "Work on communication and soft skills"
            )

        if int(float(data.get('TechnicalSkills', 0))) < 3:
            tips.append(
                "Strengthen technical skills with courses"
            )

        # ─────────────────────────────────────────────────────
        # Return Response
        # ─────────────────────────────────────────────────────
        return jsonify({

            'success': True,

            'prediction': prediction,

            'probability': round(probability, 2),

            'label':
                'PLACED ✅'
                if prediction == 1
                else 'NOT PLACED ❌',

            'tips': tips

        })

    except Exception as e:

        return jsonify({

            'success': False,
            'error': str(e)

        }), 400

# ─────────────────────────────────────────────────────────────
# Analytics Route
# ─────────────────────────────────────────────────────────────
@app.route('/analytics')
def analytics():

    best_model = MODEL_INFO.get('best_model', 'Unknown')

    return render_template(
        'analytics.html',
        model_info=MODEL_INFO,
        best_model=best_model
    )

# ─────────────────────────────────────────────────────────────
# API Route
# ─────────────────────────────────────────────────────────────
@app.route('/api/model-metrics')
def model_metrics():

    return jsonify(MODEL_INFO)

# ─────────────────────────────────────────────────────────────
# Run Flask App
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )