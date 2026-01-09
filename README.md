# VisioniX - Statistella Round 2 ML Pipeline

## 🏆 B.A.S.H Data Analytics Competition

A complete end-to-end Machine Learning pipeline for predicting **Importance Score (0-100)** for legal documents in the Statistella Round 2 competition.

## 📋 Project Overview

This project implements a robust ML pipeline using **LightGBM** with extensive feature engineering to predict document importance scores based on textual and categorical features.

### Key Features

- **Text Feature Engineering**: TF-IDF vectorization on document titles, keywords, and descriptions
- **Categorical Encoding**: Label encoding for categorical variables (state, court, case type)
- **Count-based Features**: Citation counts, keyword frequencies, topic distributions
- **Advanced Regression**: LightGBM with early stopping and hyperparameter tuning
- **Ensemble Ready**: Multiple model variants for potential stacking

## 📁 Project Structure

```
VisioniX/
├── bash-8-0-round-2/
│   ├── train.csv           # Training dataset
│   └── test.csv            # Test dataset
├── statistella_pipeline.py # Main ML pipeline
├── statistella_improved.py # Enhanced version with additional features
├── submission.csv          # Kaggle submission file
├── feature_importance.png  # Feature importance visualization
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python statistella_pipeline.py
```

### Or use the improved version

```bash
python statistella_improved.py
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Best Iteration | 813 |
| Training RMSE | ~1.55 |
| Validation RMSE | ~4.04 |

## 🔧 Tech Stack

- **Python 3.8+**
- **LightGBM** - Gradient Boosting Framework
- **Pandas** - Data Manipulation
- **Scikit-learn** - TF-IDF & Preprocessing
- **NumPy** - Numerical Computing
- **Matplotlib** - Visualization

## 📈 Feature Engineering

1. **TF-IDF Features**: Extracted from document titles, keywords, and descriptions
2. **Label Encoding**: State, court type, case type encoding
3. **Count Features**: Number of citations, keywords, topics
4. **Text Statistics**: Word counts, character lengths
5. **Frequency Features**: Keyword and topic frequencies

## 📝 Output

The pipeline generates:
- `submission.csv` - Kaggle-ready predictions with ID and Importance Score
- `feature_importance.png` - Visual representation of feature importance

## 👨‍💻 Author

**SAAIPRASATH S**

## 📄 License

This project is for the B.A.S.H Data Analytics Competition (Statistella Round 2).

---

⭐ **Star this repo if you found it helpful!**
