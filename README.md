# VisioniX - Statistella Round 2 ML Pipeline

## 🏆 B.A.S.H Data Analytics Competition

A complete end-to-end Machine Learning pipeline for predicting **Importance Score (0-100)** for legal documents in the Statistella Round 2 competition.

---

## 📋 Project Overview

This project implements a robust ML pipeline using **LightGBM** with extensive feature engineering to predict document importance scores based on textual and categorical features.

### Key Features

- **Text Feature Engineering**: TF-IDF vectorization on document titles, keywords, and descriptions
- **Categorical Encoding**: MultiLabel encoding for categorical variables
- **Count-based Features**: Text length, word counts, entity frequencies
- **Advanced Regression**: LightGBM with early stopping and hyperparameter tuning
- **Ensemble Model**: LightGBM + XGBoost ensemble (improved version)

---

## 📁 Project Structure

```
VisioniX/
├── bash-8-0-round-2/
│   ├── train.csv              # Training dataset (20,624 samples)
│   └── test.csv               # Test dataset (5,157 samples)
├── statistella_pipeline.py    # Main ML pipeline (LightGBM)
├── statistella_improved.py    # Enhanced pipeline (LightGBM + XGBoost ensemble)
├── statistella_notebook.ipynb # Kaggle notebook version
├── submission.csv             # Kaggle submission file
├── feature_importance.png     # Feature importance visualization
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Option 1: Run Basic Pipeline

```bash
python statistella_pipeline.py
```

### Option 2: Run Improved Ensemble Pipeline

```bash
python statistella_improved.py
```

### Option 3: Use Kaggle Notebook

1. Upload `statistella_notebook.ipynb` to Kaggle
2. Add the competition dataset
3. Run all cells
4. Submit the generated `submission.csv`

---

## 📊 Model Performance

| Model | Validation RMSE |
|-------|-----------------|
| LightGBM (Basic) | ~4.04 |
| LightGBM + XGBoost Ensemble | ~3.95 |

---

## 🔧 Tech Stack

- **Python 3.8+**
- **LightGBM** - Gradient Boosting Framework
- **XGBoost** - Extreme Gradient Boosting (Ensemble)
- **Pandas** - Data Manipulation
- **Scikit-learn** - TF-IDF & Preprocessing
- **NumPy** - Numerical Computing

---

## 📈 Feature Engineering Details

| Feature Type | Description | Count |
|--------------|-------------|-------|
| TF-IDF (Headline) | Unigrams & Bigrams | 500 |
| TF-IDF (Key Insights) | Unigrams & Bigrams | 1000 |
| TF-IDF (Reasoning) | Unigrams & Bigrams | 500 |
| TF-IDF (Tags) | Unigrams | 200 |
| MultiLabel (Lead Types) | Binary encoding | Variable |
| MultiLabel (Power Mentions) | Binary encoding | Variable |
| MultiLabel (Agencies) | Binary encoding | Variable |
| Count Features | Text lengths, word counts | 13 |

---

## 📝 Submission Format

The output `submission.csv` follows the required format:

```csv
id,Importance Score
21292,4.35
16024,6.45
10203,12.04
...
```

- **id**: Document identifier
- **Importance Score**: Predicted value (0-100)

---

## 👨‍💻 Author

**SAAIPRASATH S**

---

## 📄 Competition

**Statistella – B.A.S.H Round 2** | Kaggle Data Analytics Competition

---

⭐ **Star this repo if you found it helpful!**
