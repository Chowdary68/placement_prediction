# 🎓 Placement Prediction Using Machine Learning

A full-stack ML project that predicts whether a student will get placed based on their academic profile and skills.

---

## 📁 Project Structure

```
placement_prediction/
├── app.py                        # Flask web application
├── requirements.txt              # Python dependencies
│
├── data/
│   ├── generate_dataset.py       # Synthetic dataset generator
│   └── placement_data.csv        # Generated training data (1000 records)
│
├── models/
│   ├── train_model.py            # ML training pipeline (6 models)
│   ├── best_model.pkl            # Saved best model (auto-selected)
│   ├── scaler.pkl                # StandardScaler
│   ├── features.json             # Feature list
│   └── model_info.json           # All model metrics
│
├── templates/
│   ├── index.html                # Prediction UI
│   └── analytics.html           # Analytics / Charts page
│
└── static/
    ├── css/style.css
    ├── js/main.js
    └── img/                      # Auto-generated charts
        ├── model_comparison.png
        ├── confusion_matrix.png
        ├── roc_curves.png
        ├── feature_importance.png
        └── correlation_heatmap.png
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
cd data
python generate_dataset.py
```

### 3. Train models
```bash
cd models
python train_model.py
```

### 4. Run web app
```bash
python app.py
```
Open: http://localhost:5000

---

## 📊 Features Used

| Feature | Description |
|---------|-------------|
| CGPA | Cumulative Grade Point Average (5–10) |
| SSC Marks | 10th grade percentage |
| HSC Marks | 12th grade percentage |
| Internships | Number of internships completed |
| Projects | Number of projects done |
| Workshops/Certifications | Count of workshops/certs |
| Aptitude Score | Score in aptitude tests (40–100) |
| Backlogs | Number of backlogs |
| Communication Skills | Rating 1–5 |
| Technical Skills | Rating 1–5 |

---

## 🤖 ML Models Trained

| Model | Notes |
|-------|-------|
| Logistic Regression | Best baseline, high AUC |
| Decision Tree | Interpretable |
| Random Forest | Ensemble, feature importance |
| Gradient Boosting | Boosted ensemble |
| SVM | Kernel-based classification |
| KNN | Distance-based |

Best model selected automatically by AUC-ROC score.

---

## 🌐 Web Pages

- `/` — Enter student profile → get prediction + improvement tips
- `/analytics` — View all model metrics, ROC curves, feature importance
- `/api/model-metrics` — JSON API endpoint for metrics

---

## 📈 Results (Sample)

```
Model                  Acc    F1    AUC
Logistic Regression   0.995  0.996  1.000  ← Best
SVM                   0.965  0.974  0.995
Gradient Boosting     0.915  0.937  0.978
Random Forest         0.880  0.912  0.953
```

---

## 🛠 Tech Stack

- **ML**: scikit-learn, pandas, numpy
- **Viz**: matplotlib, seaborn
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, Vanilla JS
- **Persistence**: joblib

---

## 📌 Notes

- Dataset is synthetically generated for demonstration purposes
- Replace `placement_data.csv` with real college placement data for production use
- Model auto-selected based on highest AUC-ROC score
