#main.py

# --- Imports ---
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import uvicorn
import numpy as np
from scipy.sparse import hstack, csr_matrix

# --- Initialize the FastAPI app ---
app = FastAPI(
    title="SMS Fraud Detection API - Hybrid LGBM",
    description="An API to classify SMS messages using LightGBM with SYNCHRONIZED Features.",
    version="1.0.0"
)

# --- Configuration ---
NUMERICAL_FEATURES = [
    'uppercase_ratio', 'symbol_count', 'word_count',
    'url_count', 'phone_token_count', 'currency_token_count'
]
MODEL_FILE = 'final_model.joblib'
VECTORIZER_FILE = 'final_vectorizer.joblib'

# --- Pre-compiled Patterns (Synchronization & Optimization) ---
RE_EMAIL = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', re.IGNORECASE)
RE_URL = re.compile(r'(https?:\/\/\S+|www\.\S+|[a-z0-9-]+\.[a-z]{2,}\S*|wa\.me\/\S+)', re.IGNORECASE)
RE_PHONE = re.compile(r'(\+94\s?\d{9}|0\d{9})')
RE_CURRENCY_SYMBOL_FIRST = r'(?:[\$€£])\s*[\d,.\s-]+'
RE_CURRENCY_CODE_LAST = r'[\d,.\s-]+\s*(?:rs|lkr|usdt|usd|eur|gbp|pound|rupiyal|inr|aud|cad|jpy|cny|රු|රුපියල්)'
RE_CURRENCY = re.compile(f'({RE_CURRENCY_SYMBOL_FIRST}|{RE_CURRENCY_CODE_LAST})', re.IGNORECASE)
RE_NUMBER = re.compile(r'(?<![a-z])\d[\d,.]*(?![a-z])', re.IGNORECASE)

#This regex explicitly includes `<` and `>` to protect masking tags, matching the training script's logic: r'[^a-z\u0d80-\u0dff\s<>_]'
RE_NON_ALPHANUMERIC_SINHALA_SYNCHRONIZED = re.compile(r'[^a-z\u0d80-\u0dff\s<>_]')

RE_SYMBOL_COUNT = re.compile(r'[^a-zA-Z0-9\s\u0d80-\u0dff]')
DEOBFUSCATE_MAP = { '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't', '@': 'a' }

# --- Load Model/Vectorizer ---
try:
    vectorizer = joblib.load(VECTORIZER_FILE)
    model = joblib.load(MODEL_FILE)
    print(f"✅ Model ({MODEL_FILE}) and vectorizer ({VECTORIZER_FILE}) loaded successfully at startup.")
except Exception as e:
    print(f"❌ Error during model loading: {e}")
    vectorizer = None
    model = None

class SMSRequest(BaseModel):
    message: str

# --- Core Feature Synchronization Functions ---

def calculate_uppercase_ratio(text):
    text = str(text)
    total_letters = len(re.findall(r'[a-zA-Z]', text))
    uppercase_letters = len(re.findall(r'[A-Z]', text))
    return uppercase_letters / total_letters if total_letters > 0 else 0

def count_symbols(text):
    return len(RE_SYMBOL_COUNT.findall(str(text)))

def comprehensive_preprocess(text):
    """
    Applies entity masking, deobfuscation, and final cleaning,
    synchronized with the training script's logic.
    """
    text = str(text)

    # 1. Entity Masking
    text = RE_EMAIL.sub(' <email> ', text)
    text = RE_URL.sub(' <url> ', text)
    text = RE_PHONE.sub(' <phone> ', text)
    text = RE_CURRENCY.sub(' <currencyamount> ', text)
    text = RE_NUMBER.sub(' <number> ', text)

    # 2. Deobfuscation
    for char, replacement in DEOBFUSCATE_MAP.items():
        text = text.replace(char, replacement)

    # 3. Standardization and Final Cleaning
    text = text.lower()
    
    #Use the regex that explicitly preserves angle brackets `<>`
    text = RE_NON_ALPHANUMERIC_SINHALA_SYNCHRONIZED.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_words(text):
    return len(str(text).split())

def count_token(text, token):
    return str(text).count(token)

def extract_all_features(raw_text: str) -> (str, np.ndarray):
    """Master function that returns the processed text and the numerical features array."""

    feat_uppercase_ratio = calculate_uppercase_ratio(raw_text)
    feat_symbol_count = count_symbols(raw_text)

    processed_text = comprehensive_preprocess(raw_text)

    feat_word_count = count_words(processed_text)
    feat_url_count = count_token(processed_text, '<url>')
    feat_phone_token_count = count_token(processed_text, '<phone>')
    feat_currency_token_count = count_token(processed_text, '<currencyamount>')

    feature_values = [
        feat_uppercase_ratio,
        feat_symbol_count,
        feat_word_count,
        feat_url_count,
        feat_phone_token_count,
        feat_currency_token_count
    ]

    numerical_features_array = np.array(feature_values).reshape(1, -1)

    return processed_text, numerical_features_array


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "SMS Fraud Detection API is running."}

@app.post("/predict")
def predict_fraud(request: SMSRequest):
    if not model or not vectorizer:
        return {"error": "Model not loaded. Please check server logs."}

    processed_message_for_tfidf, numerical_features = extract_all_features(request.message)

    message_tfidf = vectorizer.transform([processed_message_for_tfidf])
    combined_features = hstack([message_tfidf, numerical_features]).tocsr()

    prediction = model.predict(combined_features)
    probability = model.predict_proba(combined_features)

    is_fraud = bool(prediction[0])
    confidence_score = float(probability[0][1]) if is_fraud else float(probability[0][0])

    return {
        "input_message": request.message,
        "processed_message": processed_message_for_tfidf,
        "prediction": "Fraud" if is_fraud else "Legitimate",
        "is_fraud": is_fraud,
        "confidence": f"{confidence_score:.2%}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)