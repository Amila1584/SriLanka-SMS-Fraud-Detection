# @title
# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE # RE-ENABLING SMOTE
# Install LightGBM
!pip install lightgbm
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure settings for high-quality plots
sns.set_theme(style="whitegrid")
plt.switch_backend('Agg')
pd.set_option('display.max_rows', 10)

# ==============================================================================
# 1. CONFIGURATION AND LOADING
# ==============================================================================
inputFile = 'preprocessed_data.csv'
text_column = 'message'
label_column = 'label'
RANDOM_STATE = 42
TEST_SIZE = 0.3
NUMERICAL_FEATURES = [
    'uppercase_ratio', 'symbol_count', 'word_count',
    'url_count', 'phone_token_count', 'currency_token_count'
]
PLOT_FILENAME_CM_LGBM = 'Figure_5_1_CM_LGBM_SMOTE.png'
PLOT_FILENAME_COEF_LGBM = 'Figure_5_2_LGBM_FeatureImportance_SMOTE.png'

print("--- 1. Loading Preprocessed Data ---")
df = None
try:
    df = pd.read_csv(inputFile)

    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df.dropna(subset=[text_column, label_column], inplace=True)

    print(f"✅ Successfully loaded {len(df)} unique, preprocessed data points.")
except FileNotFoundError:
    print(f"❌ CRITICAL ERROR: Preprocessed file '{inputFile}' not found. Cannot proceed.")
    exit()
except Exception as e:
    print(f"❌ Error loading or validating data: {e}")
    exit()

# ==============================================================================
# 2. DATA SPLITTING AND FEATURE TRANSFORMATION
# ==============================================================================
print("\n--- 2. Data Splitting and Vectorization ---")

y = df[label_column]
X_text = df[text_column]
X_features = df[NUMERICAL_FEATURES]

X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
    X_text, X_features, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf_train = vectorizer.fit_transform(X_text_train)
X_tfidf_test = vectorizer.transform(X_text_test)

X_train_combined = hstack([X_tfidf_train, X_features_train.values])
X_test_combined = hstack([X_tfidf_test, X_features_test.values])
print(f"✅ Combined Training Feature Shape: {X_train_combined.shape}")

# ==============================================================================
# 3. IMBALANCE HANDLING (SMOTE - Data-Level Augmentation)
# ==============================================================================
print("\n--- 3. Handling Class Imbalance with SMOTE ---")

# SMOTE is necessary to create enough positive samples for complex models to learn
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)

print(f"Original Fraud Count: {y_train.sum()}, Resampled Fraud Count: {y_train_resampled.sum()}")
print("✅ Training data balanced successfully via SMOTE.")

# ==============================================================================
# 4. MODEL TRAINING AND EVALUATION: LIGHTGBM (SMOTE-Boosted)
# ==============================================================================
print("\n--- 4. TRAINING & EVALUATION: LIGHTGBM MODEL (SMOTE-Boosted) ---")

# LightGBM requires sparse matrix to be converted to CSR format if not already
if X_train_resampled.__class__.__name__ != 'csr_matrix':
    X_train_resampled = X_train_resampled.tocsr()
    X_test_combined = X_test_combined.tocsr()

# LightGBM Classifier (No scale_pos_weight is needed here, as SMOTE balanced the data)
model_lgbm = lgb.LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    boosting_type='gbdt',
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model_lgbm.fit(X_train_resampled, y_train_resampled)
predictions_lgbm = model_lgbm.predict(X_test_combined)

print("✅ LightGBM Model Training Complete.")

# ==============================================================================
# 5. STEP 5: ACADEMIC RESULTS AND VISUALIZATION (LightGBM)
# ==============================================================================
print("\n--- 5. ACADEMIC RESULTS AND VISUALIZATION (LightGBM) ---")

# 5.1 Classification Report
print("\n## 5.1 Classification Report (LGBM SMOTE-Boosted)")
print("---------------------------------------------------------------")
print(classification_report(y_test, predictions_lgbm, target_names=['Legitimate (0)', 'Fraud (1)']))

# 5.2 Confusion Matrix Visualization (Figure 5.1)
cm_lgbm = confusion_matrix(y_test, predictions_lgbm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lgbm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Legit', 'Predicted Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud'])
plt.title('Figure 5.1: Confusion Matrix for LightGBM SMOTE-Boosted Model', fontsize=14, weight='bold')
plt.savefig(PLOT_FILENAME_CM_LGBM, dpi=300, bbox_inches='tight')
plt.show()

# 5.3 Calculating MCC and F1-Score
mcc_lgbm = matthews_corrcoef(y_test, predictions_lgbm)
fraud_recall_lgbm = cm_lgbm[1, 1] / cm_lgbm[1].sum() if cm_lgbm[1].sum() > 0 else 0
fraud_precision_lgbm = cm_lgbm[1, 1] / cm_lgbm[:, 1].sum() if cm_lgbm[:, 1].sum() > 0 else 0
fraud_f1_lgbm = 2 * (fraud_precision_lgbm * fraud_recall_lgbm) / (fraud_precision_lgbm + fraud_recall_lgbm) if (fraud_precision_lgbm + fraud_recall_lgbm) > 0 else 0

# 5.4 Feature Importance Visualization (Figure 5.2)
print("\n## 5.4 LightGBM Feature Importance (Gain)")
print("----------------------------------------------------------")

# Extracting only the importance of numerical features for visualization
numerical_feature_names = NUMERICAL_FEATURES
n_tfidf = vectorizer.get_feature_names_out().shape[0]

feature_importance = model_lgbm.feature_importances_[-len(NUMERICAL_FEATURES):]

feature_importance_df = pd.DataFrame({
    'Feature': numerical_feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.xlabel('Feature Importance (Total Gain)', fontsize=12)
plt.ylabel('Engineered Feature', fontsize=12)
plt.savefig(PLOT_FILENAME_COEF_LGBM, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nMatthews Correlation Coefficient (MCC): {mcc_lgbm:.4f}")
print(f"Fraud F1-Score: {fraud_f1_lgbm:.4f}")

print("\n--- Saved Plot Filenames ---")
print(f"✅ Visualization 1 - Confusion Matrix saved as: {PLOT_FILENAME_CM_LGBM}")
print(f"✅ Visualization 2 - Feature Importance saved as: {PLOT_FILENAME_COEF_LGBM}")