
# train_and_validate.py
# Purpose: Load, preprocess, engineer features, train, evaluate, and save the
# LightGBM model for SMS fraud detection.


# --- 0. IMPORTS ---
import pandas as pd
import numpy as np
import re
import joblib
import sys
import datetime
import warnings
import csv

# --- Sklearn and Imbalanced-Learn ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score

# --- Model ---
import lightgbm as lgb

# --- Configuration ---
TRAINING_DATA_PATH = 'final_dataset_v4.csv' # New V4 dataset path
GOLDEN_DATA_PATH = 'golden_dataset.csv' # 'golden_dataset.csv' for validation
OUTPUT_MODEL_PATH = 'final_model.joblib'
OUTPUT_VECTORIZER_PATH = 'final_vectorizer.joblib'
VALIDATION_THRESHOLD = 0.98  # 98% required accuracy on Golden Set
TEXT_COLUMN = 'message'
LABEL_COLUMN = 'label'
RANDOM_STATE = 42
TEST_SIZE = 0.3
NUMERICAL_FEATURES = [
    'uppercase_ratio', 'symbol_count', 'word_count',
    'url_count', 'phone_token_count', 'currency_token_count'
]

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- A. Logging Function with GMT+5:30 Timestamp ---
def log_message(message):
    """Prints a message with a timestamp in GMT+5:30."""
    utc_now = datetime.datetime.utcnow()
    slt_offset = datetime.timedelta(hours=5, minutes=30)
    slt_time = utc_now + slt_offset
    timestamp = slt_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp} +0530] {message}")


# B. PREPROCESSING AND FEATURE ENGINEERING FUNCTIONS


def load_and_clean_initial_data(input_file, text_col, label_col):
    """Loads CSV, assigns columns (assuming no header), and cleans initial NaNs."""
    try:
        # Load raw data and then assign columns
        with open(input_file, 'r', encoding='utf-8-sig', errors='ignore') as f:
            raw_data = list(csv.reader(f))

        df = pd.DataFrame(raw_data)
        if df.shape[1] < 2:
            raise ValueError("Dataset must contain at least two columns.")

        # Assign columns based on the expected structure [message, label, ...]
        df.columns = [text_col, label_col] + list(df.columns[2:])

        # Initial cleaning and type conversion
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
       
        rows_before_drop = len(df)
        df.dropna(subset=[label_col, text_col], inplace=True)
        df[label_col] = df[label_col].astype(int)

        log_message(f"Loaded {len(df)} messages (dropped {rows_before_drop - len(df)} corrupt rows).")
        return df

    except Exception as e:
        log_message(f"❌ Critical error during initial loading: {e}")
        return None

# --- Feature Functions (Before Preprocessing) ---
def calculate_uppercase_ratio(text):
    text = str(text)
    total_letters = len(re.findall(r'[a-zA-Z]', text))
    uppercase_letters = len(re.findall(r'[A-Z]', text))
    return uppercase_letters / total_letters if total_letters > 0 else 0

def count_symbols(text):
    # Counts non-alphanumeric, non-Sinhala symbols
    return len(re.findall(r'[^a-zA-Z0-9\s\u0d80-\u0dff]', str(text)))

# --- Comprehensive Preprocessing Function ---
def comprehensive_preprocess(text):
    """Applies entity masking, deobfuscation, and final cleaning."""
    text = str(text)

    # 1. Entity Masking
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' <email> ', text, flags=re.IGNORECASE)
    text = re.sub(r'(https?:\/\/\S+|www\.\S+|[a-z0-9-]+\.[a-z]{2,}\S*|wa\.me\/\S+)', ' <url> ', text, flags=re.IGNORECASE)
    text = re.sub(r'(\+94\s?\d{9}|0\d{9})', ' <phone> ', text)
    pattern_symbol_first = r'(?:[\$€£])\s*[\d,.\s-]+'
    pattern_code_last = r'[\d,.\s-]+\s*(?:rs|lkr|usdt|usd|eur|gbp|pound|rupiyal|inr|aud|cad|jpy|cny|රු|රුපියල්)'
    currency_pattern = f'({pattern_symbol_first}|{pattern_code_last})'
    text = re.sub(currency_pattern, ' <currencyamount> ', text, flags=re.IGNORECASE)
    # Generic number mask (avoids masking if it's part of a word)
    text = re.sub(r'(?<![a-z])\d[\d,.]*(?![a-z])', ' <number> ', text, flags=re.IGNORECASE)

    # 2. Deobfuscation (Correcting Leetspeak)
    deobfuscate_map = { '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't', '@': 'a' }
    for char, replacement in deobfuscate_map.items():
        text = text.replace(char, replacement)

    # 3. Standardization and Final Cleaning
    text = text.lower()
    text = re.sub(r'[^a-z\u0d80-\u0dff\s<>_]', ' ', text) # Remove remaining symbols
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    return text

# --- Feature Functions (After Preprocessing) ---
def count_words(text):
    return len(str(text).split())

def count_token(text, token):
    return str(text).count(token)


# C. DATA PREPARATION WORKFLOW


def prepare_data(df_raw, is_golden=False):
    """Applies all feature engineering and preprocessing steps to a DataFrame."""
    df = df_raw.copy()
   
    log_message("--- Calculating Structural Features (Pre-cleaning) ---")
    df['uppercase_ratio'] = df[TEXT_COLUMN].apply(calculate_uppercase_ratio)
    df['symbol_count'] = df[TEXT_COLUMN].apply(count_symbols)

    log_message("--- Applying Comprehensive Preprocessing (Masking & Cleaning) ---")
    df['message_processed'] = df[TEXT_COLUMN].apply(comprehensive_preprocess)

    log_message("--- Calculating Token Features (Post-cleaning) ---")
    df['word_count'] = df['message_processed'].apply(count_words)
    df['url_count'] = df['message_processed'].apply(lambda x: count_token(x, '<url>'))
    df['phone_token_count'] = df['message_processed'].apply(lambda x: count_token(x, '<phone>'))
    df['currency_token_count'] = df['message_processed'].apply(lambda x: count_token(x, '<currencyamount>'))

    if not is_golden:
        # Deduplication only needed for the main training set
        initial_count = len(df)
        df.drop_duplicates(subset=['message_processed'], inplace=True, keep='first')
        log_message(f"Removed {initial_count - len(df)} duplicate messages.")

    # Final cleanup: Drop the original column and rename the processed column
    df.drop(columns=[TEXT_COLUMN], inplace=True)
    df.rename(columns={'message_processed': TEXT_COLUMN}, inplace=True)
   
    # Ensure all numerical features are correctly typed and handle NaNs
    for col in NUMERICAL_FEATURES:
         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
   
    return df


# D. MAIN EXECUTION FLOW


if __name__ == "__main__":
   
    # --- 1. Load and Prepare Data ---
    log_message("--- 1. Loading and Preparing Datasets ---")
    df_raw = load_and_clean_initial_data(TRAINING_DATA_PATH, TEXT_COLUMN, LABEL_COLUMN)
    df_golden_raw = load_and_clean_initial_data(GOLDEN_DATA_PATH, TEXT_COLUMN, LABEL_COLUMN)

    if df_raw is None or df_golden_raw is None:
        sys.exit(1)
       
    df_train_final = prepare_data(df_raw, is_golden=False)
    df_golden_final = prepare_data(df_golden_raw, is_golden=True)
    log_message(f"Final Training Data Size: {len(df_train_final)}")
    log_message(f"Final Golden Data Size: {len(df_golden_final)}")

    # --- 2. Splitting Data ---
    log_message("--- 2. Splitting Data and Defining Features ---")
    y = df_train_final[LABEL_COLUMN]
    X_text = df_train_final[TEXT_COLUMN]
    X_features = df_train_final[NUMERICAL_FEATURES]

    X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
        X_text, X_features, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # --- 3. TF-IDF Vectorization and Combination ---
    log_message("--- 3. Vectorizing Text and Combining Features ---")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_tfidf_train = vectorizer.fit_transform(X_text_train)
    X_tfidf_test = vectorizer.transform(X_text_test)

    X_train_combined = hstack([X_tfidf_train, X_features_train.values]).tocsr()
    X_test_combined = hstack([X_tfidf_test, X_features_test.values]).tocsr()
    log_message(f"Combined Training Feature Shape: {X_train_combined.shape}")

    # --- 4. SMOTE Oversampling ---
    log_message("--- 4. Applying SMOTE to Training Data ---")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)
    log_message(f"Original Fraud Count: {y_train.sum()}, Resampled Fraud Count: {y_train_resampled.sum()}")

    # --- 5. Train LightGBM Model ---
    log_message("--- 5. Training LightGBM Model ---")
    model_lgbm = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        n_estimators=100, # Using default for quick training, consider increasing later
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1 # Suppress minor logging
    )
    model_lgbm.fit(X_train_resampled, y_train_resampled)
    log_message("✅ LightGBM model training complete.")

    # --- 6. Validate Against Golden Dataset ---
    log_message("--- 6. Validating Candidate Model Against Golden Dataset ---")
    X_golden_text = df_golden_final[TEXT_COLUMN]
    X_golden_features = df_golden_final[NUMERICAL_FEATURES]
    y_golden_true = df_golden_final[LABEL_COLUMN].tolist()
   
    # Transform Golden Data features
    X_golden_tfidf = vectorizer.transform(X_golden_text)
    X_golden_combined = hstack([X_golden_tfidf, X_golden_features.values]).tocsr()

    y_golden_pred = model_lgbm.predict(X_golden_combined)

    correct_predictions = sum(1 for i in range(len(y_golden_true)) if y_golden_true[i] == y_golden_pred[i])
    total_predictions = len(y_golden_true)
    success_rate = correct_predictions / total_predictions if total_predictions > 0 else 0

    log_message(f"Correctly classified: {correct_predictions} out of {total_predictions}")
    log_message(f"New model success rate on golden data: {success_rate:.2%}")
    log_message(f"Required success rate to pass validation: {VALIDATION_THRESHOLD:.2%}")

    # --- 7. Evaluate Model Performance (Test Set) ---
    log_message("--- 7. Calculating Detailed Performance Metrics (Test Set) ---")
    predictions = model_lgbm.predict(X_test_combined)

    # Focus on the Fraud Class (pos_label=1)
    precision_fraud = precision_score(y_test, predictions, pos_label=1)
    recall_fraud = recall_score(y_test, predictions, pos_label=1)
    f1_fraud = f1_score(y_test, predictions, pos_label=1)
    mcc = matthews_corrcoef(y_test, predictions)

    log_message("--- PERFORMANCE SUMMARY (Test Set) ---")
    log_message(
        f"Fraud (1) Metrics: [Precision: {precision_fraud:.4f}, Recall: {recall_fraud:.4f}, F1: {f1_fraud:.4f}]"
    )
    log_message(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    # --- 8. Save Model if Validation Passes ---
    if success_rate >= VALIDATION_THRESHOLD:
        log_message(" VALIDATION SUCCESS: Model passed threshold.")
        log_message("Saving model and vectorizer for deployment...")
        joblib.dump(model_lgbm, OUTPUT_MODEL_PATH)
        joblib.dump(vectorizer, OUTPUT_VECTORIZER_PATH)
        log_message(f" Files saved: '{OUTPUT_MODEL_PATH}' and '{OUTPUT_VECTORIZER_PATH}'")
        sys.exit(0)
    else:
        log_message(" VALIDATION FAILED: Model did not meet success rate threshold.")
        sys.exit(1)