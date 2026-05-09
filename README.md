# 🏦 Credit Score Checker

A machine learning web application that predicts customer creditworthiness using a **Random Forest** classifier trained on the German Credit Dataset.

---

## 📌 Project Overview

Banks and financial institutions need to assess whether a loan applicant is likely to repay or default. This project builds an end-to-end credit risk prediction system — from data preprocessing and model training to a live web app where customers can be assessed instantly.

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.7440 |
| Precision (Good) | 0.8045 |
| Recall (Good) | 0.8229 |
| F1-Score (Good) | 0.8136 |
| ROC-AUC | 0.7587 |
| **CV-AUC** | **0.8873 ± 0.0175** |

### Classification Report

```
 Accuracy  : 0.7440
  Bad  Recall    : 0.6133
  Bad  F1-Score  : 0.5897
  Good F1-Score  : 0.8140
  ROC-AUC        : 0.7723

              precision    recall  f1-score   support

         Bad       0.57      0.61      0.59        75
        Good       0.83      0.80      0.81       175

    accuracy                           0.74       250
   macro avg       0.70      0.71      0.70       250
weighted avg       0.75      0.74      0.75       250

confusion_matrix:
[[ 46  29]
 [ 35 140]]
```

---

## 📊 Dataset

- **Source:** German Credit Dataset
- **Records:** 1,000 customers
- **Class Distribution:** 700 Good (70%) / 300 Bad (30%)
- **Features:** 9 input features

| Feature | Description |
|---------|-------------|
| Age | Customer age in years |
| Sex | Male / Female |
| Job | Skill level (0 = unskilled, 3 = highly skilled) |
| Housing | Own / Rent / Free |
| Saving accounts | Little / Moderate / Quite Rich / Rich |
| Checking account | Little / Moderate / Rich |
| Credit amount | Loan amount in Euros (€) |
| Duration | Loan duration in months |
| Purpose | Car / Education / Business / etc. |

---

## 🤖 Model Details

### Approach
- **Algorithm:** Random Forest Classifier (100 trees)
- **Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Class Weights:** Balanced (extra protection for Bad class)
- **Decision Threshold:** 0.47 (tuned for best Bad-class F1)

### Why SMOTE?
The dataset is imbalanced (70/30). Without SMOTE, the model would heavily favour the majority "Good" class and miss most "Bad" applicants — which is the most costly mistake in credit risk.

### Threshold Tuning

The default 0.50 threshold can be adjusted to control the precision/recall tradeoff:

| Threshold | Bad Recall | Bad F1 | Good F1 | Best For |
|-----------|-----------|--------|---------|----------|
| 0.40 | 53% | 0.559 | 0.828 | Protecting Good applicants |
| **0.47** | **60%** | **0.596** | **0.795** | **Best overall (recommended)** |
| 0.50 | 64% | 0.575 | 0.787 | Balanced default |
| 0.55 | 68% | 0.560 | 0.748 | Aggressive risk screening |
| 0.60 | 77% | 0.592 | 0.737 | Strictest credit control |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/credit-scoring.git
cd credit-scoring
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---


## 📦 Requirements

```
flask
scikit-learn
imbalanced-learn
joblib
numpy
pandas
gunicorn
```

Install all at once:
```bash
pip install flask scikit-learn imbalanced-learn joblib numpy pandas gunicorn
```

---

## 🖥️ Web App Preview

The app provides a clean form UI where you enter customer details and get an instant prediction:

- ✅ **Customer is SAFE** — green result card
- ⚠️ **Customer is RISKY** — red result card
- 📊 Confidence score, probability gauge, and risk percentage displayed

### Sample Low-Risk Customer (Test Input)

| Field | Value |
|-------|-------|
| Age | 45 |
| Sex | Male |
| Job | 2 — Skilled |
| Housing | Own |
| Saving Account | Rich |
| Checking Account | Moderate |
| Credit Amount | €2,500 |
| Duration | 12 months |
| Purpose | Radio / TV |

---

## 🔧 Customisation

### Change the Decision Threshold

Open `app.py` and update line:
```python
THRESHOLD = 0.47   # ← change this value
```

| Value | Effect |
|-------|--------|
| Lower (e.g. 0.40) | More applicants approved, fewer Bad caught |
| Higher (e.g. 0.60) | Fewer approvals, more Bad caught |

### Retrain the Model

Run all cells in `credit_scoring_model.ipynb` — it will regenerate `model.pkl` and `encoders.json` automatically.

---

## 📈 Future Improvements

- [ ] Add XGBoost / LightGBM comparison
- [ ] Cost-sensitive learning with custom misclassification weights
- [ ] SHAP explainability — show which features drove the decision
- [ ] Batch prediction via CSV upload
- [ ] Admin dashboard with prediction history
- [ ] User authentication for bank staff

---

## 👤 Author

**Rakshitha Poshetty**  
B.Tech / Data Science Project  
[GitHub](https://github.com/rakshitha2706) · [LinkedIn](https://www.linkedin.com/in/rakshitha-poshetty-4a4269357)

---

## 📄 License

This project is for educational purposes. Dataset sourced from the UCI Machine Learning Repository.
