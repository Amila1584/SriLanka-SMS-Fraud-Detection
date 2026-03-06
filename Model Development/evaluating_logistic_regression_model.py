# @title
# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE

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
PLOT_FILENAME_CM_LR = 'Figure_4_1_CM_LR_SMOTE.png'
PLOT_FILENAME_COEF_LR = 'Figure_4_2_LR_FeatureImportance_SMOTE.png'

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
print("\n--- 2. Splitting Data and Vectorization ---")

y = df[label_column]
X_text = df[text_column]
X_features = df[NUMERICAL_FEATURES]

# Data Splitting (Stratified)
X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
    X_text, X_features, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"Dataset split: {len(y_train)} training samples, {len(y_test)} testing samples.")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_tfidf_train = vectorizer.fit_transform(X_text_train)
X_tfidf_test = vectorizer.transform(X_text_test)

# Feature Combination
X_train_combined = hstack([X_tfidf_train, X_features_train.values])
X_test_combined = hstack([X_tfidf_test, X_features_test.values])
print(f"Combined Training Feature Shape: {X_train_combined.shape}")

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
# 4. MODEL TRAINING AND EVALUATION
# ==============================================================================
print("\n--- 4. Training SMOTE-Boosted Logistic Regression Model ---")

# Logistic Regression (No class_weight needed as SMOTE balanced the data)
model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE)
model.fit(X_train_resampled, y_train_resampled)
print("✅ SMOTE-Boosted Model Training Complete.")

# Prediction and Metric Calculation
predictions = model.predict(X_test_combined)

# ==============================================================================
# 5. STEP 5: ACADEMIC RESULTS AND VISUALIZATION
# ==============================================================================
print("\n--- 5. Academic Results and Visualization ---")

# 5.1 Classification Report
print("\n## 5.1 Classification Report (SMOTE-Boosted LR Baseline)")
print("---------------------------------------------------------------")
print(classification_report(y_test, predictions, target_names=['Legitimate (0)', 'Fraud (1)']))

# 5.2 Confusion Matrix Visualization (Figure 3.3)
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Legit', 'Predicted Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud'])
# Title removed per request
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig(PLOT_FILENAME_CM_LR, dpi=300, bbox_inches='tight')
print(f"✅ Confusion Matrix saved as: {PLOT_FILENAME_CM_LR}")
plt.show()

# 5.3 Calculating MCC and F1-Score
mcc = matthews_corrcoef(y_test, predictions)
fraud_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
fraud_precision = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() > 0 else 0
fraud_f1 = 2 * (fraud_precision * fraud_recall) / (fraud_precision + fraud_recall) if (fraud_precision + fraud_recall) > 0 else 0


# 5.4 Feature Importance Visualization (Figure 3.4)
print("\n## 5.4 Numeric Feature Importance Analysis (LR Coefficients)")
print("----------------------------------------------------------------")

# Extract the coefficients for the engineered numerical features
coefficients = model.coef_[0][-len(NUMERICAL_FEATURES):]

feature_coef = pd.DataFrame({
    'Feature': NUMERICAL_FEATURES,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_coef, palette='viridis')
# Title removed per request
plt.xlabel('Coefficient Value (Impact on Predicting Fraud)', fontsize=12)
plt.ylabel('Engineered Feature', fontsize=12)
plt.savefig(PLOT_FILENAME_COEF_LR, dpi=300, bbox_inches='tight')
print(f"✅ Coefficient Plot saved as: {PLOT_FILENAME_COEF_LR}")
plt.show()

print("\n--- Final Metrics ---")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"Fraud F1-Score: {fraud_f1:.4f}")

print("\n--- Saved Plot Filenames ---")
print(f"✅ Visualization 1 - Confusion Matrix saved as: {PLOT_FILENAME_CM_LR}")
print(f"✅ Visualization 2 - Feature Importance saved as: {PLOT_FILENAME_COEF_LR}")