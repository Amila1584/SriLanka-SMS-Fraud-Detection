# @title
# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, make_scorer, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure settings for high-quality plots
sns.set_theme(style="whitegrid")
plt.switch_backend('Agg')
pd.set_option('display.max_rows', 10)

# ==============================================================================
# 1. CONFIGURATION AND LOADING (Same as before)
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
# Updated Figure Naming to Section 6
PLOT_FILENAME_CM_LGBM_T = 'Figure_6_1_CM_LGBM_Tuned.png'
PLOT_FILENAME_COEF_LGBM_T = 'Figure_6_2_LGBM_FeatureImportance_Tuned.png'
PLOT_FILENAME_TUNING = 'Figure_6_3_LGBM_Tuning_Process.png'

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

vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_features=20000 )
X_tfidf_train = vectorizer.fit_transform(X_text_train)
X_tfidf_test = vectorizer.transform(X_text_test)

X_train_combined = hstack([X_tfidf_train, X_features_train.values])
X_test_combined = hstack([X_tfidf_test, X_features_test.values])
print(f"✅ Combined Training Feature Shape: {X_train_combined.shape}")

# ==============================================================================
# 3. IMBALANCE HANDLING (SMOTE - Data-Level Augmentation)
# ==============================================================================
print("\n--- 3. Handling Class Imbalance with SMOTE ---")

smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)

print(f"Original Fraud Count: {y_train.sum()}, Resampled Fraud Count: {y_train_resampled.sum()}")
print("✅ Training data balanced successfully via SMOTE.")

# LightGBM requires sparse matrix to be converted to CSR format if not already
if X_train_resampled.__class__.__name__ != 'csr_matrix':
    X_train_resampled = X_train_resampled.tocsr()
    X_test_combined = X_test_combined.tocsr()


# ==============================================================================
# 6. STEP 6: HYPERPARAMETER TUNING AND FINAL EVALUATION
# ==============================================================================
print("\n--- [6] HYPERPARAMETER TUNING AND FINAL EVALUATION ---")

# Define parameter grid for Randomized Search
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 50],
    'max_depth': [5, 10, 15],
    'min_child_samples': [10, 20, 50],
    'subsample': [0.7, 0.8, 0.9]
}

# Use F1-Score for the positive class (Fraud) as the tuning metric
f1_scorer = make_scorer(f1_score, pos_label=1)

# Initialize LGBM model
lgbm_base = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1 # Suppress warnings during training
)

# Randomized Search CV
random_search = RandomizedSearchCV(
    estimator=lgbm_base,
    param_distributions=param_grid,
    n_iter=10, # Number of parameter settings that are sampled
    scoring=f1_scorer,
    cv=3, # 3-fold cross-validation
    verbose=0,
    random_state=RANDOM_STATE
)

# Fit the randomized search on the SMOTE-resampled training data
random_search.fit(X_train_resampled, y_train_resampled)
best_model = random_search.best_estimator_
print("✅ Hyperparameter Tuning Complete.")

# 6.1 Hyperparameter Tuning Visualization (Figure 6.3)
results_df = pd.DataFrame(random_search.cv_results_)
plt.figure(figsize=(10, 6))
sns.lineplot(x=results_df['param_learning_rate'], y=results_df['mean_test_score'], marker='o', label='F1 Score vs. Learning Rate')
plt.xlabel('Learning Rate (Hyperparameter)', fontsize=12)
plt.ylabel('Mean Cross-Validation F1-Score', fontsize=12)
plt.savefig(PLOT_FILENAME_TUNING, dpi=300, bbox_inches='tight')
print(f"✅ Figure 6.3 (Tuning Visualization) saved as: {PLOT_FILENAME_TUNING}")
plt.show()


# 6.2 Final Prediction and Evaluation
predictions_tuned = best_model.predict(X_test_combined)

# 6.3 Classification Report
print("\n## 6.3 Final Classification Report (Tuned LightGBM)")
print("---------------------------------------------------------")
print(classification_report(y_test, predictions_tuned, target_names=['Legitimate (0)', 'Fraud (1)']))

# 6.4 Final Metrics
cm_tuned = confusion_matrix(y_test, predictions_tuned)
mcc_tuned = matthews_corrcoef(y_test, predictions_tuned)
fraud_recall_tuned = cm_tuned[1, 1] / cm_tuned[1].sum() if cm_tuned[1].sum() > 0 else 0
fraud_precision_tuned = cm_tuned[1, 1] / cm_tuned[:, 1].sum() if cm_tuned[:, 1].sum() > 0 else 0
fraud_f1_tuned = 2 * (fraud_precision_tuned * fraud_recall_tuned) / (fraud_precision_tuned + fraud_recall_tuned) if (fraud_precision_tuned + fraud_recall_tuned) > 0 else 0

print(f"\nMatthews Correlation Coefficient (MCC): {mcc_tuned:.4f}")
print(f"Fraud F1-Score: {fraud_f1_tuned:.4f}")

# 6.5 Final Confusion Matrix Visualization (Figure 6.1)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Legit', 'Predicted Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud'])
plt.savefig(PLOT_FILENAME_CM_LGBM_T, dpi=300, bbox_inches='tight')
print(f"✅ Figure 6.1 (Tuned CM) saved as: {PLOT_FILENAME_CM_LGBM_T}")
plt.show()

print("\n--- Saved Plot Filenames ---")
print(f"✅ Visualization 1 - Confusion Matrix saved as: {PLOT_FILENAME_CM_LGBM_T}")
print(f"✅ Visualization 2 - Feature Importance saved as: {PLOT_FILENAME_COEF_LGBM_T}")
print(f"✅ Visualization 3 - Hyperparameter tuning visualization saved as: {PLOT_FILENAME_TUNING}")