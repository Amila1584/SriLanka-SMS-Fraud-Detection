# @title
# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Assuming SMOTE is available, though typically from imblearn
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression # Needed for setup/comparison

# Configure settings for high-quality plots for dissertation output
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
# ***CORRECTED FIGURE NAMING CONVENTION***
PLOT_FILENAME_CM_RF = 'Figure_3_1_CM_RF_SMOTE.png'
PLOT_FILENAME_COEF_RF = 'Figure_3_2_RF_FeatureImportance_SMOTE.png'

print("--- 1. Loading Preprocessed Data ---")
df = None
try:
    df = pd.read_csv(inputFile)

    # Ensure numerical features are correctly typed and handle any NaNs
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df.dropna(subset=[text_column, label_column], inplace=True)

    print(f"✅ Successfully loaded {len(df)} unique, preprocessed data points.")
except FileNotFoundError:
    print(f"❌ CRITICAL ERROR: Preprocessed file '{inputFile}' not found. Cannot proceed with training.")
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

# Data Splitting (Stratified)
X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
    X_text, X_features, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf_train = vectorizer.fit_transform(X_text_train)
X_tfidf_test = vectorizer.transform(X_text_test)

# Feature Combination
X_train_combined = hstack([X_tfidf_train, X_features_train.values])
X_test_combined = hstack([X_tfidf_test, X_features_test.values])
print("✅ Data split and features vectorized.")

# ==============================================================================
# 3. IMBALANCE HANDLING (SMOTE - Data-Level Augmentation)
# ==============================================================================
print("\n--- 3. Handling Class Imbalance with SMOTE ---")

# Apply SMOTE to the training data
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)

print(f"Original Fraud Count: {y_train.sum()}, Resampled Fraud Count: {y_train_resampled.sum()}")
print("✅ Training data balanced successfully via SMOTE.")

# ==============================================================================
# 4. MODEL TRAINING AND EVALUATION (Random Forest - SMOTE-Boosted)
# ==============================================================================
print("\n--- 4. Training SMOTE-Boosted Random Forest Model ---")

# Random Forest Classifier (No class_weight needed as SMOTE balanced the data)
model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model_rf.fit(X_train_resampled, y_train_resampled)
print("✅ SMOTE-Boosted Random Forest Training Complete.")

# Prediction and Metric Calculation
predictions_rf = model_rf.predict(X_test_combined)

# ==============================================================================
# 5. STEP 5: ACADEMIC RESULTS AND VISUALIZATION
# ==============================================================================
print("\n--- 5. Academic Results and Visualization (Figures 4.1 & 4.2) ---")

# 5.1 Classification Report
print("\n## 5.1 Classification Report (SMOTE-Boosted Random Forest)")
print("----------------------------------------------------------------------")
print(classification_report(y_test, predictions_rf, target_names=['Legitimate (0)', 'Fraud (1)']))

# 5.2 Confusion Matrix Visualization (Figure 3.3)
cm_rf = confusion_matrix(y_test, predictions_rf)
plt.figure(figsize=(8, 6))
# Using 'Blues' as requested for visual consistency
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Legit', 'Predicted Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud'])
# Title removed per request
plt.savefig(PLOT_FILENAME_CM_RF, dpi=300, bbox_inches='tight')
plt.show()

# 5.3 Calculating MCC and F1-Score
mcc_rf = matthews_corrcoef(y_test, predictions_rf)
fraud_recall_rf = cm_rf[1, 1] / cm_rf[1].sum() if cm_rf[1].sum() > 0 else 0
fraud_precision_rf = cm_rf[1, 1] / cm_rf[:, 1].sum() if cm_rf[:, 1].sum() > 0 else 0
fraud_f1_rf = 2 * (fraud_precision_rf * fraud_recall_rf) / (fraud_precision_rf + fraud_recall_rf) if (fraud_precision_rf + fraud_recall_rf) > 0 else 0


print(f"\nMatthews Correlation Coefficient (MCC): {mcc_rf:.4f}")
print(f"Fraud F1-Score: {fraud_f1_rf:.4f}")

# 5.4 Feature Importance Visualization (Figure 3.4)
print("\n## 5.4 Random Forest Feature Importance (Gini Importance)")
print("----------------------------------------------------------")

feature_importance = model_rf.feature_importances_[-len(NUMERICAL_FEATURES):]
feature_importance_df = pd.DataFrame({
    'Feature': NUMERICAL_FEATURES,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
# Title removed per request
plt.xlabel('Feature Importance (Gini Index)', fontsize=12)
plt.ylabel('Engineered Feature', fontsize=12)
plt.savefig(PLOT_FILENAME_COEF_RF, dpi=300, bbox_inches='tight')
plt.show()

print("\n--- Saved Plot Filenames ---")
print(f"✅ Visualization 1 - Confusion Matrix saved as: {PLOT_FILENAME_CM_RF}")
print(f"✅ Visualization 2 - Feature Importance saved as: {PLOT_FILENAME_COEF_RF}")