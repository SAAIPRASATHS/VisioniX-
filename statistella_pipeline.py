"""
Statistella Round 2 - Importance Score Prediction Pipeline
============================================================
B.A.S.H Data Analytics Competition - Kaggle Notebook

Goal: Predict Importance Score (0-100) for legal documents using
text features (TF-IDF) and categorical features with LightGBM.
"""

import pandas as pd
import numpy as np
import warnings
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA LOADING
# ============================================================
print("=" * 60)
print("STATISTELLA ROUND 2 - ML PIPELINE")
print("=" * 60)
print("\n[*] Loading datasets...")

train = pd.read_csv('bash-8-0-round-2/train.csv')
test = pd.read_csv('bash-8-0-round-2/test.csv')

print("[+] Train shape:", train.shape)
print("[+] Test shape:", test.shape)
print("\nColumns:", list(train.columns))

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Target distribution
print("\nTarget 'Importance Score' Statistics:")
print(train['Importance Score'].describe())

# Missing values
print("\nMissing Values in Train:")
print(train.isnull().sum())

# ============================================================
# 3. DATA CLEANING
# ============================================================
print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Text columns to process
text_cols = ['Headline', 'Reasoning', 'Key Insights', 'Tags']
# List-like categorical columns
list_cols = ['Lead Types', 'Power Mentions', 'Agencies']

def clean_text(text):
    """Clean and normalize text."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def parse_list_column(value):
    """Parse semicolon or comma separated values."""
    if pd.isna(value) or value == '':
        return []
    # Split by semicolon or comma
    items = re.split(r'[;,]', str(value))
    return [item.strip().lower() for item in items if item.strip()]

# Clean text columns
for col in text_cols:
    train[col + '_clean'] = train[col].apply(clean_text)
    test[col + '_clean'] = test[col].apply(clean_text)
    print("[+] Cleaned", col)

# Parse list columns
for col in list_cols:
    train[col + '_list'] = train[col].apply(parse_list_column)
    test[col + '_list'] = test[col].apply(parse_list_column)
    print("[+] Parsed", col)

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# --- 4.1 TF-IDF Features ---
print("\n[*] Creating TF-IDF features...")

# Combine train and test for consistent vectorization
all_headlines = pd.concat([train['Headline_clean'], test['Headline_clean']])
all_insights = pd.concat([train['Key Insights_clean'], test['Key Insights_clean']])
all_reasoning = pd.concat([train['Reasoning_clean'], test['Reasoning_clean']])
all_tags = pd.concat([train['Tags_clean'], test['Tags_clean']])

# TF-IDF Vectorizers
tfidf_headline = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=3)
tfidf_insights = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=3)
tfidf_reasoning = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=3)
tfidf_tags = TfidfVectorizer(max_features=200, ngram_range=(1, 1), min_df=3)

# Fit and transform
tfidf_headline.fit(all_headlines)
tfidf_insights.fit(all_insights)
tfidf_reasoning.fit(all_reasoning)
tfidf_tags.fit(all_tags)

train_headline_tfidf = tfidf_headline.transform(train['Headline_clean'])
test_headline_tfidf = tfidf_headline.transform(test['Headline_clean'])

train_insights_tfidf = tfidf_insights.transform(train['Key Insights_clean'])
test_insights_tfidf = tfidf_insights.transform(test['Key Insights_clean'])

train_reasoning_tfidf = tfidf_reasoning.transform(train['Reasoning_clean'])
test_reasoning_tfidf = tfidf_reasoning.transform(test['Reasoning_clean'])

train_tags_tfidf = tfidf_tags.transform(train['Tags_clean'])
test_tags_tfidf = tfidf_tags.transform(test['Tags_clean'])

print("[+] Headline TF-IDF:", train_headline_tfidf.shape[1], "features")
print("[+] Key Insights TF-IDF:", train_insights_tfidf.shape[1], "features")
print("[+] Reasoning TF-IDF:", train_reasoning_tfidf.shape[1], "features")
print("[+] Tags TF-IDF:", train_tags_tfidf.shape[1], "features")

# --- 4.2 MultiLabel Encoding for List Columns ---
print("\n[*] Encoding categorical list columns...")

# Combine for fitting
all_lead_types = train['Lead Types_list'].tolist() + test['Lead Types_list'].tolist()
all_power_mentions = train['Power Mentions_list'].tolist() + test['Power Mentions_list'].tolist()
all_agencies = train['Agencies_list'].tolist() + test['Agencies_list'].tolist()

mlb_lead = MultiLabelBinarizer(sparse_output=True)
mlb_power = MultiLabelBinarizer(sparse_output=True)
mlb_agency = MultiLabelBinarizer(sparse_output=True)

mlb_lead.fit(all_lead_types)
mlb_power.fit(all_power_mentions)
mlb_agency.fit(all_agencies)

train_lead_enc = mlb_lead.transform(train['Lead Types_list'])
test_lead_enc = mlb_lead.transform(test['Lead Types_list'])

train_power_enc = mlb_power.transform(train['Power Mentions_list'])
test_power_enc = mlb_power.transform(test['Power Mentions_list'])

train_agency_enc = mlb_agency.transform(train['Agencies_list'])
test_agency_enc = mlb_agency.transform(test['Agencies_list'])

print("[+] Lead Types:", train_lead_enc.shape[1], "unique labels")
print("[+] Power Mentions:", train_power_enc.shape[1], "unique labels")
print("[+] Agencies:", train_agency_enc.shape[1], "unique labels")

# --- 4.3 Count-based Features ---
print("\n[*] Creating count-based features...")

def create_count_features(df):
    """Create numerical features from text and list columns."""
    features = pd.DataFrame()
    
    # Text length features
    features['headline_len'] = df['Headline'].fillna('').apply(len)
    features['reasoning_len'] = df['Reasoning'].fillna('').apply(len)
    features['insights_len'] = df['Key Insights'].fillna('').apply(len)
    features['tags_len'] = df['Tags'].fillna('').apply(len)
    
    # Word count features
    features['headline_words'] = df['Headline'].fillna('').apply(lambda x: len(str(x).split()))
    features['reasoning_words'] = df['Reasoning'].fillna('').apply(lambda x: len(str(x).split()))
    features['insights_words'] = df['Key Insights'].fillna('').apply(lambda x: len(str(x).split()))
    
    # Entity count features
    features['lead_types_count'] = df['Lead Types_list'].apply(len)
    features['power_mentions_count'] = df['Power Mentions_list'].apply(len)
    features['agencies_count'] = df['Agencies_list'].apply(len)
    
    # Has entity flags
    features['has_lead_types'] = (features['lead_types_count'] > 0).astype(int)
    features['has_power_mentions'] = (features['power_mentions_count'] > 0).astype(int)
    features['has_agencies'] = (features['agencies_count'] > 0).astype(int)
    
    return features

train_counts = create_count_features(train)
test_counts = create_count_features(test)
print("[+] Created", train_counts.shape[1], "count-based features")

# --- 4.4 Combine All Features ---
print("\n[*] Combining all features...")

# Convert count features to sparse
train_counts_sparse = csr_matrix(train_counts.values)
test_counts_sparse = csr_matrix(test_counts.values)

# Stack all features horizontally
X_train = hstack([
    train_headline_tfidf,
    train_insights_tfidf,
    train_reasoning_tfidf,
    train_tags_tfidf,
    train_lead_enc,
    train_power_enc,
    train_agency_enc,
    train_counts_sparse
])

X_test = hstack([
    test_headline_tfidf,
    test_insights_tfidf,
    test_reasoning_tfidf,
    test_tags_tfidf,
    test_lead_enc,
    test_power_enc,
    test_agency_enc,
    test_counts_sparse
])

y_train = train['Importance Score'].values

print("[+] Total training features:", X_train.shape[1])
print("[+] Training samples:", X_train.shape[0])
print("[+] Test samples:", X_test.shape[0])

# ============================================================
# 5. MODELING WITH LIGHTGBM
# ============================================================
print("\n" + "=" * 60)
print("TRAINING LIGHTGBM MODEL")
print("=" * 60)

# Train/Validation Split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print("\nSplit sizes:")
print("   Training:", X_tr.shape[0])
print("   Validation:", X_val.shape[0])

# LightGBM parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': 10,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 2000,
    'early_stopping_rounds': 100,
    'verbose': -1,
    'random_state': 42
}

print("\nTraining with parameters:")
for key, value in list(lgb_params.items())[:6]:
    print("  ", key, ":", value)

# Create LightGBM datasets
lgb_train = lgb.Dataset(X_tr, label=y_tr)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

# Train model
print("\n[*] Training model...")
model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
)

# Validation predictions
val_preds = model.predict(X_val)
val_preds = np.clip(val_preds, 0, 100)  # Clip to valid range

val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
print("[SUCCESS] Validation RMSE:", round(val_rmse, 4))

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Top 20)")
print("=" * 60)

# Create feature names
feature_names = (
    [f'headline_tfidf_{i}' for i in range(train_headline_tfidf.shape[1])] +
    [f'insights_tfidf_{i}' for i in range(train_insights_tfidf.shape[1])] +
    [f'reasoning_tfidf_{i}' for i in range(train_reasoning_tfidf.shape[1])] +
    [f'tags_tfidf_{i}' for i in range(train_tags_tfidf.shape[1])] +
    [f'lead_{l}' for l in mlb_lead.classes_] +
    [f'power_{p}' for p in mlb_power.classes_] +
    [f'agency_{a}' for a in mlb_agency.classes_] +
    list(train_counts.columns)
)

importance = model.feature_importance(importance_type='gain')
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False).head(20)

for idx, row in importance_df.iterrows():
    print("  ", row['feature'][:40].ljust(40), ":", round(row['importance'], 2))

# Save importance plot
try:
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'].values[::-1], importance_df['importance'].values[::-1])
    plt.xlabel('Importance (Gain)')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print("\n[+] Feature importance plot saved to 'feature_importance.png'")
except Exception as e:
    print("\n[!] Could not save feature importance plot:", str(e))

# ============================================================
# 7. GENERATE PREDICTIONS FOR TEST SET
# ============================================================
print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

test_preds = model.predict(X_test)
test_preds = np.clip(test_preds, 0, 100)  # Ensure values are in valid range

print("[+] Generated", len(test_preds), "predictions")
print("[+] Prediction range: [", round(test_preds.min(), 2), ",", round(test_preds.max(), 2), "]")

# ============================================================
# 8. CREATE SUBMISSION FILE
# ============================================================
print("\n" + "=" * 60)
print("CREATING SUBMISSION FILE")
print("=" * 60)

submission = pd.DataFrame({
    'id': test['id'],
    'Importance Score': test_preds
})

submission.to_csv('submission.csv', index=False)
print("[+] Submission saved to 'submission.csv'")
print("[+] Shape:", submission.shape)
print("\nPreview:")
print(submission.head(10))

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)
print(f"""
Summary:
  - Training samples: {X_train.shape[0]}
  - Test samples: {X_test.shape[0]}
  - Total features: {X_train.shape[1]}
  - Validation RMSE: {val_rmse:.4f}
  - Model: LightGBM with early stopping
  - Best iteration: {model.best_iteration}
""")
