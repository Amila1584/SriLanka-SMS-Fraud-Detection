# @title
# ==============================================================================
# prepare_dataset.py
# Purpose: Load, clean, engineer ALL features, and save data for reuse.
# Ensures output column order is: message, label, feature1, feature2, ...
# ==============================================================================
import pandas as pd
import csv
import re
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
inputFile = 'final_dataset_v4.csv'
outputFile = 'preprocessed_data.csv'
text_column = 'message'
label_column = 'label'
# ---------------------

def load_and_clean_initial_data(input_file, text_col, label_col):
    """Loads CSV, handles columns, and cleans initial NaNs."""
    print("--- 1. Loading and Initial Cleaning ---")

    try:
        # Load raw data assuming no header and then assign columns
        with open(input_file, 'r', encoding='utf-8-sig', errors='ignore') as f:
            raw_data = list(csv.reader(f))

            df = pd.DataFrame(raw_data)
            if df.shape[1] < 2:
                raise ValueError("Dataset must contain at least two columns.")

            # CRITICAL FIX: Explicit Column Assignment [message, label, ...]
            df.columns = [text_col, label_col] + list(df.columns[2:])

            # Initial cleaning and type conversion
            df[label_col] = pd.to_numeric(df[label_col], errors='coerce')

            rows_before_drop = len(df)
            df.dropna(subset=[label_col, text_col], inplace=True)
            df[label_col] = df[label_col].astype(int)

            print(f"Successfully loaded {len(df)} messages (dropped {rows_before_drop - len(df)} corrupt rows).")
            return df

    except Exception as e:
        print(f"❌ Critical error during initial loading: {e}")
        return None

# --- FEATURE ENGINEERING FUNCTIONS (BEFORE PREPROCESSING) ---

def calculate_uppercase_ratio(text):
    """Calculates the ratio of uppercase letters to total letters."""
    text = str(text)
    total_letters = len(re.findall(r'[a-zA-Z]', text))
    uppercase_letters = len(re.findall(r'[A-Z]', text))
    return uppercase_letters / total_letters if total_letters > 0 else 0

def count_symbols(text):
    """Counts non-alphanumeric, non-Sinhala symbols (punctuation, etc.)."""
    return len(re.findall(r'[^a-zA-Z0-9\s\u0d80-\u0dff]', str(text)))

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
    text = re.sub(r'(?<![a-z])\d[\d,.]*(?![a-z])', ' <number> ', text, flags=re.IGNORECASE)

    # 2. Deobfuscation (Correcting Leetspeak)
    deobfuscate_map = { '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't', '@': 'a' }
    for char, replacement in deobfuscate_map.items():
        text = text.replace(char, replacement)

    # 3. Standardization and Final Cleaning
    text = text.lower()
    text = re.sub(r'[^a-z\u0d80-\u0dff\s<>_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- FEATURE ENGINEERING FUNCTIONS (AFTER PREPROCESSING) ---

def count_words(text):
    """Counts the total number of tokens (words) in the preprocessed text."""
    return len(str(text).split())

def count_token(text, token):
    """Counts the occurrences of a specific masked token."""
    return str(text).count(token)

# --- MAIN EXECUTION ---
df_clean = load_and_clean_initial_data(inputFile, text_column, label_column)

if df_clean is not None:
    print("\n--- 2. Feature Engineering & Preprocessing ---")

    # --- FEATURES CALCULATED ON RAW TEXT (Before lowercasing) ---
    df_clean['uppercase_ratio'] = df_clean[text_column].apply(calculate_uppercase_ratio)
    df_clean['symbol_count'] = df_clean[text_column].apply(count_symbols)

    # Apply text preprocessing (Stores the result in a temporary column)
    df_clean['message_processed'] = df_clean[text_column].apply(comprehensive_preprocess)

    # --- FEATURES CALCULATED ON PROCESSED TEXT ---
    df_clean['word_count'] = df_clean['message_processed'].apply(count_words)
    df_clean['url_count'] = df_clean['message_processed'].apply(lambda x: count_token(x, '<url>'))
    df_clean['phone_token_count'] = df_clean['message_processed'].apply(lambda x: count_token(x, '<phone>'))
    df_clean['currency_token_count'] = df_clean['message_processed'].apply(lambda x: count_token(x, '<currencyamount>'))

    print("✅ All new linguistic and structural features calculated.")

    print("\n--- 3. Removing Duplicates (Post-Preprocessing) ---")
    initial_count = len(df_clean)
    # Deduplicate based on the cleaned text content
    df_clean.drop_duplicates(subset=['message_processed'], inplace=True, keep='first')
    print(f"✅ Removed {initial_count - len(df_clean)} duplicate messages.")
    print(f"Final unique dataset size: {len(df_clean)}.")

    # Final cleanup: Drop the original column and rename the processed column
    df_clean.drop(columns=[text_column], inplace=True)
    df_clean.rename(columns={'message_processed': text_column}, inplace=True)

    # --- 4. CRITICAL: ENFORCE FINAL COLUMN ORDER ---
    feature_cols = [col for col in df_clean.columns if col not in [text_column, label_column]]
    final_order = [text_column, label_column] + feature_cols
    df_final = df_clean[final_order]

    print(f"\n✅ Final DataFrame column order enforced: {final_order[:2]}...")

    print("\n--- 5. Final Feature Set Preview and Saving ---")
    print(df_final.head().to_markdown())

    # Save the processed DataFrame for reuse
    df_final.to_csv(outputFile, index=False, encoding='utf-8')
    print(f"✅ Preprocessed data saved to: {outputFile}")