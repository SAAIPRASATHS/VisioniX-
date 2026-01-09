import pandas as pd
import numpy as np
import warnings
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

print("=" * 60)
print("STATISTELLA ROUND 2 - IMPROVED PIPELINE")
print("=" * 60)

print("\n[*] Loading datasets...")
train = pd.read_csv('bash-8-0-round-2/train.csv')
test = pd.read_csv('bash-8-0-round-2/test.csv')
print("[+] Train shape:", train.shape)
print("[+] Test shape:", test.shape)

print("\n[*] Cleaning data...")

text_cols = ['Headline', 'Reasoning', 'Key Insights', 'Tags']
list_cols = ['Lead Types', 'Power Mentions', 'Agencies']

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def parse_list_column(value):
    if pd.isna(value) or value == '':
        return []
    items = re.split(r'[;,]', str(value))
    return [item.strip().lower() for item in items if item.strip()]

for col in text_cols:
    train[col + '_clean'] = train[col].apply(clean_text)
    test[col + '_clean'] = test[col].apply(clean_text)

for col in list_cols:
    train[col + '_list'] = train[col].apply(parse_list_column)
    test[col + '_list'] = test[col].apply(parse_list_column)

print("[+] Data cleaned")

print("\n[*] Creating features...")

all_headlines = pd.concat([train['Headline_clean'], test['Headline_clean']])
all_insights = pd.concat([train['Key Insights_clean'], test['Key Insights_clean']])
all_reasoning = pd.concat([train['Reasoning_clean'], test['Reasoning_clean']])
all_tags = pd.concat([train['Tags_clean'], test['Tags_clean']])

tfidf_headline = TfidfVectorizer(max_features=600, ngram_range=(1, 2), min_df=2)
tfidf_insights = TfidfVectorizer(max_features=1200, ngram_range=(1, 2), min_df=2)
tfidf_reasoning = TfidfVectorizer(max_features=600, ngram_range=(1, 2), min_df=2)
tfidf_tags = TfidfVectorizer(max_features=300, ngram_range=(1, 1), min_df=2)

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

print("[+] TF-IDF features:", train_headline_tfidf.shape[1] + train_insights_tfidf.shape[1] + 
      train_reasoning_tfidf.shape[1] + train_tags_tfidf.shape[1])

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

print("[+] Multilabel features:", train_lead_enc.shape[1] + train_power_enc.shape[1] + train_agency_enc.shape[1])

def create_enhanced_features(df):
    features = pd.DataFrame()
    
    features['headline_len'] = df['Headline'].fillna('').apply(len)
    features['reasoning_len'] = df['Reasoning'].fillna('').apply(len)
    features['insights_len'] = df['Key Insights'].fillna('').apply(len)
    features['tags_len'] = df['Tags'].fillna('').apply(len)
    
    features['headline_words'] = df['Headline'].fillna('').apply(lambda x: len(str(x).split()))
    features['reasoning_words'] = df['Reasoning'].fillna('').apply(lambda x: len(str(x).split()))
    features['insights_words'] = df['Key Insights'].fillna('').apply(lambda x: len(str(x).split()))
    features['tags_words'] = df['Tags'].fillna('').apply(lambda x: len(str(x).split()))
    
    features['headline_sentences'] = df['Headline'].fillna('').apply(lambda x: len(re.split(r'[.!?]', str(x))))
    features['reasoning_sentences'] = df['Reasoning'].fillna('').apply(lambda x: len(re.split(r'[.!?]', str(x))))
    features['insights_sentences'] = df['Key Insights'].fillna('').apply(lambda x: len(re.split(r'[.!?]', str(x))))
    
    features['headline_avg_word_len'] = df['Headline'].fillna('').apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0)
    features['reasoning_avg_word_len'] = df['Reasoning'].fillna('').apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0)
    
    features['lead_types_count'] = df['Lead Types_list'].apply(len)
    features['power_mentions_count'] = df['Power Mentions_list'].apply(len)
    features['agencies_count'] = df['Agencies_list'].apply(len)
    
    features['total_entities'] = features['lead_types_count'] + features['power_mentions_count'] + features['agencies_count']
    
    features['has_lead_types'] = (features['lead_types_count'] > 0).astype(int)
    features['has_power_mentions'] = (features['power_mentions_count'] > 0).astype(int)
    features['has_agencies'] = (features['agencies_count'] > 0).astype(int)
    features['has_all_entities'] = ((features['has_lead_types'] + features['has_power_mentions'] + features['has_agencies']) == 3).astype(int)
    
    features['reasoning_to_headline_ratio'] = features['reasoning_len'] / (features['headline_len'] + 1)
    features['insights_to_reasoning_ratio'] = features['insights_len'] / (features['reasoning_len'] + 1)
    
    features['log_reasoning_len'] = np.log1p(features['reasoning_len'])
    features['log_lead_types_count'] = np.log1p(features['lead_types_count'])
    
    return features

train_counts = create_enhanced_features(train)
test_counts = create_enhanced_features(test)
print("[+] Enhanced count features:", train_counts.shape[1])

train_counts_sparse = csr_matrix(train_counts.values)
test_counts_sparse = csr_matrix(test_counts.values)

X_train = hstack([
    train_headline_tfidf, train_insights_tfidf, train_reasoning_tfidf, train_tags_tfidf,
    train_lead_enc, train_power_enc, train_agency_enc,
    train_counts_sparse
])

X_test = hstack([
    test_headline_tfidf, test_insights_tfidf, test_reasoning_tfidf, test_tags_tfidf,
    test_lead_enc, test_power_enc, test_agency_enc,
    test_counts_sparse
])

y_train = train['Importance Score'].values
print("[+] Total features:", X_train.shape[1])

print("\n" + "=" * 60)
print("TRAINING WITH 5-FOLD CROSS-VALIDATION")
print("=" * 60)

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 48,
    'max_depth': 8,
    'min_child_samples': 30,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'reg_alpha': 0.2,
    'reg_lambda': 0.2,
    'n_estimators': 3000,
    'verbose': -1,
    'random_state': 42
}

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.03,
    'max_depth': 6,
    'min_child_weight': 30,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.2,
    'reg_lambda': 0.2,
    'n_estimators': 3000,
    'verbosity': 0,
    'random_state': 42
}

lgb_oof = np.zeros(len(y_train))
xgb_oof = np.zeros(len(y_train))
lgb_test_preds = np.zeros(X_test.shape[0])
xgb_test_preds = np.zeros(X_test.shape[0])

lgb_cv_scores = []
xgb_cv_scores = []

print("\n[*] Training LightGBM & XGBoost ensemble...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
    
    X_tr = X_train[train_idx]
    X_val = X_train[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]
    
    lgb_train_data = lgb.Dataset(X_tr, label=y_tr)
    lgb_val_data = lgb.Dataset(X_val, label=y_val, reference=lgb_train_data)
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train_data,
        valid_sets=[lgb_val_data],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=500)
        ]
    )
    
    lgb_val_pred = lgb_model.predict(X_val)
    lgb_oof[val_idx] = lgb_val_pred
    lgb_test_preds += lgb_model.predict(X_test) / N_FOLDS
    lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
    lgb_cv_scores.append(lgb_rmse)
    print(f"  LightGBM RMSE: {lgb_rmse:.4f}")
    
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_oof[val_idx] = xgb_val_pred
    xgb_test_preds += xgb_model.predict(X_test) / N_FOLDS
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
    xgb_cv_scores.append(xgb_rmse)
    print(f"  XGBoost RMSE: {xgb_rmse:.4f}")

print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS")
print("=" * 60)

lgb_mean_rmse = np.mean(lgb_cv_scores)
xgb_mean_rmse = np.mean(xgb_cv_scores)
lgb_oof_rmse = np.sqrt(mean_squared_error(y_train, lgb_oof))
xgb_oof_rmse = np.sqrt(mean_squared_error(y_train, xgb_oof))

print(f"\nLightGBM:")
print(f"  CV Mean RMSE: {lgb_mean_rmse:.4f} (+/- {np.std(lgb_cv_scores):.4f})")
print(f"  OOF RMSE: {lgb_oof_rmse:.4f}")

print(f"\nXGBoost:")
print(f"  CV Mean RMSE: {xgb_mean_rmse:.4f} (+/- {np.std(xgb_cv_scores):.4f})")
print(f"  OOF RMSE: {xgb_oof_rmse:.4f}")

print("\n[*] Optimizing ensemble weights...")
best_weight = 0.5
best_rmse = float('inf')

for w in np.arange(0.3, 0.8, 0.05):
    blended_oof = w * lgb_oof + (1 - w) * xgb_oof
    blend_rmse = np.sqrt(mean_squared_error(y_train, blended_oof))
    if blend_rmse < best_rmse:
        best_rmse = blend_rmse
        best_weight = w

print(f"  Best LightGBM weight: {best_weight:.2f}")
print(f"  Best XGBoost weight: {1 - best_weight:.2f}")

ensemble_oof = best_weight * lgb_oof + (1 - best_weight) * xgb_oof
ensemble_rmse = np.sqrt(mean_squared_error(y_train, ensemble_oof))

print(f"\n[SUCCESS] ENSEMBLE OOF RMSE: {ensemble_rmse:.4f}")
print(f"[+] Improvement over baseline (4.0296): {4.0296 - ensemble_rmse:.4f}")

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

final_preds = best_weight * lgb_test_preds + (1 - best_weight) * xgb_test_preds
final_preds = np.clip(final_preds, 0, 100)

print(f"[+] Generated {len(final_preds)} predictions")
print(f"[+] Prediction range: [{final_preds.min():.2f}, {final_preds.max():.2f}]")

submission = pd.DataFrame({
    'id': test['id'],
    'Importance Score': final_preds
})

submission.to_csv('submission_improved.csv', index=False)
print("[+] Saved to 'submission_improved.csv'")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)
print(f"""
Summary:
  - Features: {X_train.shape[1]}
  - LightGBM CV RMSE: {lgb_mean_rmse:.4f}
  - XGBoost CV RMSE: {xgb_mean_rmse:.4f}
  - Ensemble OOF RMSE: {ensemble_rmse:.4f}
  - Baseline was: 4.0296
  - Improvement: {4.0296 - ensemble_rmse:.4f}
""")
