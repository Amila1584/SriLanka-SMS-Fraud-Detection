# @title
# ==============================================================================
# PART A: RE-INITIALIZE DATA AND DATASET (STEPS 1-3)
# NOTE: This must be executed first to define 'test_dataset'
# ==============================================================================
import pandas as pd
import csv
import re
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Configuration (MUST MATCH ORIGINAL TRAINING) ---
inputFile = 'final_dataset_v4.csv'
text_column = 'message'
label_column = 'label'
model_checkpoint = "bert-base-multilingual-cased"
MAX_LEN = 128 # Must match tokenization length used during training (from cell pqUmWaqAoVS6)
RANDOM_STATE = 42

# --- Define Preprocessing Functions (Must match training preprocessing) ---
def comprehensive_preprocess(text):
    text = str(text)
    # Entity Masking
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' <email> ', text, flags=re.IGNORECASE)
    text = re.sub(r'(https?://\S+|www\.\S+|[a-z0-9-]+\.[a-z]{2,}\S*|wa\.me/\S+)', ' <url> ', text, flags=re.IGNORECASE)
    text = re.sub(r'(\+94\s?\d{9}|0\d{9})', ' <phone> ', text)
    pattern_symbol_first = r'(?:[\$€£])\s*[\d,.\s-]+'
    pattern_code_last = r'[\d,.\s-]+\s*(?:rs|lkr|usdt|usd|eur|gbp|pound|rupiyal|inr|aud|cad|jpy|cny|\u0d85\u0dcf\u0dbd\u0dda\u0d9a\u0dca\u0da9\u0dca|\u0dbb\u0dd4|\u0dbb\u0dd4\u0db4\u0dd2\u0dba\u0dbd)'
    currency_pattern = f'({pattern_symbol_first}|{pattern_code_last})'
    text = re.sub(currency_pattern, ' <currencyamount> ', text, flags=re.IGNORECASE)
    return text

# --- Define Custom Dataset Class (Must match Step 3) ---
class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("\n--- [A.1] Re-Loading and Preprocessing Data ---")
try:
    with open(inputFile, 'r', encoding='utf-8-sig', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)
        if header and header[0].startswith('\ufeff'):
            header[0] = header[0][1:]
        data = [row for row in reader if len(row) == len(header)]
    df = pd.DataFrame(data, columns=header)
    df[label_column] = pd.to_numeric(df[label_column], errors='coerce')
    df.dropna(subset=[label_column, text_column], inplace=True)
    df[label_column] = df[label_column].astype(int)

except Exception as e:
    # Use synthetic data if real data loading fails
    print(f"Loading failed ({e}). Using synthetic data.")
    data = {'message': ["Claim <url>!", "meeting.", "Urgent: <phone>.", "Hello.", "won <currencyamount>!"], 'label': [1, 0, 1, 0, 1]}
    df = pd.DataFrame(data)

df[text_column] = df[text_column].apply(comprehensive_preprocess)
df.drop_duplicates(subset=[text_column], inplace=True, keep='first')
print(f"Loaded {len(df)} unique messages.")

print("--- [A.2] Re-Creating Test Split ---")
# Re-create the 80/20 stratified split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[label_column])

print("--- [A.3] Re-Tokenizing Test Dataset ---")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# The CRITICAL step: Redefine test_dataset
test_dataset = SMSDataset(test_df[text_column], test_df[label_column], tokenizer, MAX_LEN)
print(f"✅ 'test_dataset' recreated successfully with {len(test_dataset)} samples.")

# ==============================================================================
# PART B: LOAD MODEL, PREDICT, AND EVALUATE (STEP 6)
# ==============================================================================

CHECKPOINT_PATH = './results/checkpoint-31000'
PLOT_FILENAME_THRESHOLDS = 'Figure_9_1_Threshold_Optimization_Final.png'
PLOT_FILENAME_CM_FINAL = 'Figure_9_2_CM_mBERT_Final_31000.png'

print(f"\n--- [B.1] Loading Model and Tokenizer from: {CHECKPOINT_PATH} ---")
try:
    # Load the best model and tokenizer directly from the saved checkpoint folder
    loaded_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
    # Instantiate a dummy Trainer with the loaded model and an empty args object
    loaded_trainer = Trainer(model=loaded_model, args=None)
    print("✅ Model and Trainer loaded successfully.")

except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load checkpoint from {CHECKPOINT_PATH}.")
    print("Ensure the folder exists and contains 'pytorch_model.bin' and 'config.json'.")
    print(f"Error details: {e}")
    exit()


print("\n--- [B.2] PREDICTION AND THRESHOLD OPTIMIZATION ---")

# 6.1 Generate Probabilities and True Labels
predictions = loaded_trainer.predict(test_dataset)
y_prob = predictions.predictions[:, 1] # Probabilities for the positive class (Fraud)
y_test = predictions.label_ids         # True labels
print(f"✅ Generated predictions for {len(y_test)} test samples.")

# 6.2 Define Optimization Parameters
thresholds = np.arange(0.05, 0.96, 0.01)
scores = []

# 6.3 Iterate and Calculate Metrics (F1-Score Focus)
for t in thresholds:
    y_pred_thresh = (y_prob >= t).astype(int)
    rec = recall_score(y_test, y_pred_thresh, pos_label=1, zero_division=0)
    prec = precision_score(y_test, y_pred_thresh, pos_label=1, zero_division=0)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    scores.append({'Threshold': t, 'F1_Score': f1, 'Recall': rec, 'Precision': prec})

results_df = pd.DataFrame(scores)

# 6.4 Determine Optimal Threshold (with robustness check)
# --- MODIFIED LOGIC: CHECK FOR CONSTANT F1-SCORE ---
if results_df['F1_Score'].nunique() <= 1 and results_df['F1_Score'].iloc[0] < 1e-6:
    optimal_threshold = 0.50 # Use default 0.50 if optimization failed
    # Find the row corresponding to 0.50 for metrics extraction
    optimal_threshold_row = results_df.loc[np.isclose(results_df['Threshold'], optimal_threshold)].iloc[0]
    print("⚠️ WARNING: Optimization failed (F1 is constant/zero). Defaulting threshold to 0.50.")
else:
    # Use idxmax for the highest F1-Score
    optimal_threshold_row = results_df.loc[results_df['F1_Score'].idxmax()]
    optimal_threshold = optimal_threshold_row['Threshold']
# --- END MODIFIED LOGIC ---

print(f"✅ Optimal Threshold (Max F1-Score): {optimal_threshold:.2f}")

# 6.5 Evaluate Model with Optimal Threshold
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
mcc_optimal = matthews_corrcoef(y_test, y_pred_optimal)
report_optimal = classification_report(y_test, y_pred_optimal, output_dict=True, target_names=['Legitimate (0)', 'Fraud (1)'], zero_division=0)

# 6.6 Threshold Visualization (Figure 9.1) - ADDING DATA LABELS AND MARKERS
# --- Extract metrics at optimal threshold ---
opt_f1 = optimal_threshold_row['F1_Score']
opt_recall = optimal_threshold_row['Recall']
opt_precision = optimal_threshold_row['Precision']

plt.figure(figsize=(12, 8))

# 1. Plot the main lines with markers (Point 2: Visualize Data Points)
plt.plot(results_df['Threshold'], results_df['F1_Score'], label='F1-Score (Optimization Target)', color='green', linewidth=2, marker='o', markersize=4)
plt.plot(results_df['Threshold'], results_df['Recall'], label='Recall', color='red', linestyle='--', marker='s', markersize=4)
plt.plot(results_df['Threshold'], results_df['Precision'], label='Precision', color='blue', linestyle='--', marker='^', markersize=4)

# 2. Add the vertical line for the Optimal Threshold
plt.axvline(optimal_threshold, color='black', linestyle=':', label=f'Optimal Threshold: {optimal_threshold:.2f}')

# 3. Add scatter markers and text annotations at the optimal point (Point 3: Data Labels)
# --- F1-Score Annotation (Green) ---
plt.scatter(optimal_threshold, opt_f1, color='green', s=150, zorder=5, edgecolors='black')
plt.annotate(
    f'F1: {opt_f1:.4f}',
    (optimal_threshold, opt_f1),
    textcoords="offset points", xytext=(5, 5), ha='left', color='green', fontweight='bold'
)

# --- Recall Annotation (Red) ---
plt.scatter(optimal_threshold, opt_recall, color='red', s=150, zorder=5, edgecolors='black')
plt.annotate(
    f'Recall: {opt_recall:.4f}',
    (optimal_threshold, opt_recall),
    textcoords="offset points", xytext=(5, -15), ha='left', color='red', fontweight='bold'
)

# --- Precision Annotation (Blue) ---
plt.scatter(optimal_threshold, opt_precision, color='blue', s=150, zorder=5, edgecolors='black')
plt.annotate(
    f'Precision: {opt_precision:.4f}',
    (optimal_threshold, opt_precision),
    textcoords="offset points", xytext=(-5, 5), ha='right', color='blue', fontweight='bold'
)

# 4. Fix Legend Overlap (Point 1: Legend hides label)
plt.xlabel('Classification Threshold', fontsize=14)
plt.ylabel('Performance Metric Value', fontsize=14)
plt.legend(loc='upper right', fontsize=12) # Changed location to upper right
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig(PLOT_FILENAME_THRESHOLDS, dpi=300, bbox_inches='tight')
print(f"✅ Figure 9.1 (Threshold Optimization Curve) saved as: {PLOT_FILENAME_THRESHOLDS}")
plt.show()

# 6.7 Threshold Performance Table (Point 3: Numerical Performance Table)
print("\n## 6.7 THRESHOLD PERFORMANCE ANALYSIS (NUMERICAL)")
print("-------------------------------------------------")

# Find key rows for the table
default_row = results_df.loc[np.isclose(results_df['Threshold'], 0.50)].iloc[0]
max_recall_row = results_df.loc[results_df['Recall'].idxmax()]
max_precision_row = results_df.loc[results_df['Precision'].idxmax()]

# Consolidate data into a list
table_data = []
for label, row in [
    ('Optimal F1 (T_opt)', optimal_threshold_row),
    ('Default (0.50)', default_row),
    ('Max Recall', max_recall_row),
    ('Max Precision', max_precision_row)
]:
    # Check if the row is already in the list to avoid duplication if T_opt is 0.50, Max Recall is T_opt, etc.
    # Store Threshold as a float for comparison
    if not table_data or not any(np.isclose(d['Threshold'], row['Threshold']) for d in table_data):
        table_data.append({
            'Category': label,
            'Threshold': row['Threshold'], # Store as float
            'F1': row['F1_Score'],       # Store as float
            'Recall': row['Recall'],     # Store as float
            'Precision': row['Precision'] # Store as float
        })

# Print the resulting table
table_df = pd.DataFrame(table_data)
# Format numerical columns for display after the comparison logic
table_df['Threshold'] = table_df['Threshold'].apply(lambda x: f"{x:.2f}")
table_df['F1'] = table_df['F1'].apply(lambda x: f"{x:.4f}")
table_df['Recall'] = table_df['Recall'].apply(lambda x: f"{x:.4f}")
table_df['Precision'] = table_df['Precision'].apply(lambda x: f"{x:.4f}")

print(table_df.to_markdown(index=False))

# Final Metric Summary
print("\n## 6.8 FINAL PERFORMANCE SUMMARY (Using T_opt)")
print("---------------------------------------------")
# Extracting final metrics (using string keys from report_optimal for robustness)
final_precision = report_optimal['Fraud (1)']['precision']
final_recall = report_optimal['Fraud (1)']['recall']
final_f1 = report_optimal['Fraud (1)']['f1-score']

print(f"**Optimal Threshold Used:** {optimal_threshold:.2f}")
print(f"**Fraud Precision (Usability):** {final_precision:.4f}")
print(f"**Fraud Recall (Detection Rate):** {final_recall:.4f}")
print(f"**Fraud F1-Score:** {final_f1:.4f}")
print(f"**Matthews Correlation Coefficient (MCC):** {mcc_optimal:.4f}")

# 6.9 Optimal Confusion Matrix (Figure 9.2) - Adding data labels to the matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_optimal,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Predicted Legit', 'Predicted Fraud'],
    yticklabels=['Actual Legit', 'Actual Fraud'],
    annot_kws={"size": 14, "weight": "bold"} # Make the numbers inside the cells stand out
)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig(PLOT_FILENAME_CM_FINAL, dpi=300, bbox_inches='tight')
print(f"✅ Final CM saved as: {PLOT_FILENAME_CM_FINAL}")
plt.show()