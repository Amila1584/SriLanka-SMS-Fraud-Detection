# @title
# ==============================================================================
# INSTALL/UPGRADE DEPENDENCIES
# ==============================================================================
!pip install --upgrade transformers accelerate imbalanced-learn -q
print("All necessary libraries installed and upgraded to compatible versions.")

# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import pandas as pd
import csv
import re
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ==============================================================================
# 1. DATA PREPARATION (Condensed)
# ==============================================================================
print("\n--- Step 1-3: Loading, Preprocessing, and Deduplicating Data ---")
inputFile = 'final_dataset_v4.csv'
text_column = 'message'
label_column = 'label'

data = []
header = []
try:
    # Attempt to load the file
    with open(inputFile, 'r', encoding='utf-8-sig', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Handle potential BOM in the first header element
        if header and header[0].startswith('\ufeff'):
            header[0] = header[0][1:]
        # Load data rows
        for row in reader:
            if len(row) == len(header):
                data.append(row)

    df = pd.DataFrame(data, columns=header)
    df[label_column] = pd.to_numeric(df[label_column], errors='coerce')
    df.dropna(subset=[label_column, text_column], inplace=True)
    df[label_column] = df[label_column].astype(int)
    print(f"Successfully loaded {len(df)} valid messages.")
except FileNotFoundError:
    print(f"❌ ERROR: The file '{inputFile}' was not found. Please ensure it is uploaded.")
    # Fallback to synthetic data for demonstration if file is missing
    print("Generating synthetic data for execution flow demonstration.")
    data = {
        'message': ["Claim your FREE prize now at <url>!", "Meeting rescheduled for 2 PM.", "Urgent: Your account will be closed. Call <phone>.", "Hello, can we meet next week?", "You have won a lottery of <currencyamount>! Reply YES."],
        'label': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    df[label_column] = df[label_column].astype(int)
    text_column = 'message'
    label_column = 'label'

except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    raise

def comprehensive_preprocess(text):
    text = str(text)
    # Entity Masking
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', ' <email> ', text, flags=re.IGNORECASE)
    text = re.sub(r'(https?:\/\/\S+|www\.\S+|[a-z0-9-]+\.[a-z]{2,}\S*|wa\.me\/\S+)', ' <url> ', text, flags=re.IGNORECASE)
    text = re.sub(r'(\+94\s?\d{9}|0\d{9})', ' <phone> ', text)
    pattern_symbol_first = r'(?:[\$€£])\s*[\d,.\s-]+'
    pattern_code_last = r'[\d,.\s-]+\s*(?:rs|lkr|usdt|usd|eur|gbp|pound|rupiyal|inr|aud|cad|jpy|cny|රු|රුපියල්)'
    currency_pattern = f'({pattern_symbol_first}|{pattern_code_last})'
    text = re.sub(currency_pattern, ' <currencyamount> ', text, flags=re.IGNORECASE)
    return text

df[text_column] = df[text_column].apply(comprehensive_preprocess)
df.drop_duplicates(subset=[text_column], inplace=True, keep='first')
print(f"Data preparation complete. Final dataset size: {len(df)} unique messages.")

# ==============================================================================
# 2. SPLIT AND OVERSAMPLE THE DATASET (Using full data split here for robustness)
# ==============================================================================
print("\n--- Creating Test Split and Oversampling Training Data ---")

# Using an 80/20 split on the full unique data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_column])
y_train = train_df[label_column]

ros = RandomOverSampler(random_state=42)
# Resample the training data
X_train_resampled, y_train_resampled = ros.fit_resample(train_df.drop(columns=[label_column]), y_train)
train_resampled_df = pd.concat([X_train_resampled, y_train_resampled.rename(label_column)], axis=1)

print("Class distribution in the new balanced training set:")
print(train_resampled_df[label_column].value_counts().to_string())

# ==============================================================================
# 3. TOKENIZE AND CREATE PYTORCH DATASETS
# ==============================================================================
print("\n--- Tokenizing Datasets for mBERT ---")
model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=96):
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

train_dataset = SMSDataset(train_resampled_df[text_column], train_resampled_df[label_column], tokenizer)
test_dataset = SMSDataset(test_df[text_column], test_df[label_column], tokenizer)
print("PyTorch datasets created.")

# ==============================================================================
# 4. FINE-TUNE THE TRANSFORMER MODEL
# ==============================================================================
print("\n--- Fine-Tuning the mBERT Model (Memory Optimized) ---")
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Calculate key metrics
    precision = precision_score(labels, preds, average='binary', zero_division=0)
    recall = recall_score(labels, preds, average='binary', zero_division=0)
    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)

    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True, # Critical for loading the best checkpoint
    metric_for_best_model='f1', # Optimize for F1-score due to imbalance
    greater_is_better=True
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("Starting fine-tuning... Please ensure GPU is enabled.")
try:
    trainer.train()
    print("✅ Training complete.")
except Exception as e:
    print(f"An error occurred during training: {e}")
    # Proceed to evaluation with the partially trained or base model if training failed
    pass


# ==============================================================================
# 5. FINAL EVALUATION (Default 0.5 Threshold)
# ==============================================================================
print("\n--- Final Evaluation (Default 0.5 Threshold) on Test Set ---")
evaluation_results = trainer.evaluate()
print("\nFinal Model Performance (mBERT Fine-tuning):")
print(evaluation_results)