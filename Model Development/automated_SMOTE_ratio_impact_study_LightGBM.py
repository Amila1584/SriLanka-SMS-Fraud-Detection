# @title Automated SMOTE Ratio Impact Study
# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure settings
sns.set_theme(style="whitegrid")
plt.switch_backend('Agg')

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

print("--- 1. Loading Preprocessed Data ---")
df = pd.read_csv(inputFile)
for col in NUMERICAL_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
df.dropna(subset=[text_column, label_column], inplace=True)

# ==============================================================================
# 2. DATA SPLITTING AND VECTORIZATION
# ==============================================================================
y = df[label_column]
X_text = df[text_column]
X_features = df[NUMERICAL_FEATURES]

X_text_train, X_text_test, X_features_train, X_features_test, y_train, y_test = train_test_split(
    X_text, X_features, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

vectorizer = TfidfVectorizer(ngram_range=(1, 2),max_features=20000 )
X_tfidf_train = vectorizer.fit_transform(X_text_train)
X_tfidf_test = vectorizer.transform(X_text_test)

X_train_combined = hstack([X_tfidf_train, X_features_train.values]).tocsr()
X_test_combined = hstack([X_tfidf_test, X_features_test.values]).tocsr()

# ==============================================================================
# 3. SMOTE RATIO IMPACT STUDY
# ==============================================================================
# Define ratios: sampling_strategy is (minority_count / majority_count)
# 1.0 means 100% (Balanced), 0.1 means minority becomes 10% of majority
ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] 
results = []

print(f"\n--- Running Sensitivity Analysis on SMOTE Ratios ---")
print(f"{'Ratio':<10} | {'Recall':<8} | {'Precision':<10} | {'F1-Score':<8} | {'MCC':<8}")
print("-" * 55)

for ratio in ratios:
    if ratio == 0.0:
        X_res, y_res = X_train_combined, y_train
        label_text = "No SMOTE"
    else:
        sm = SMOTE(sampling_strategy=ratio, random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train_combined, y_train)
        label_text = f"{int(ratio*100)}%"

    # Train LightGBM
    clf = lgb.LGBMClassifier(objective='binary', n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    clf.fit(X_res, y_res)
    
    # Predict on UNCHANGED imbalanced test set
    preds = clf.predict(X_test_combined)
    
    # Calculate Metrics for Fraud class (1)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    
    results.append({
        'SMOTE Ratio': label_text,
        'Recall': rec,
        'Precision': prec,
        'F1-Score': f1,
        'MCC': mcc
    })
    
    print(f"{label_text:<10} | {rec:<8.4f} | {prec:<10.4f} | {f1:<8.4f} | {mcc:<8.4f}")

# ==============================================================================
# 4. VISUALIZATION OF IMPACT
# ==============================================================================
res_df = pd.DataFrame(results)

plt.figure(figsize=(12, 6))
plt.plot(res_df['SMOTE Ratio'], res_df['Recall'], marker='o', label='Fraud Recall', linewidth=2, color='red')
plt.plot(res_df['SMOTE Ratio'], res_df['Precision'], marker='s', label='Fraud Precision', linewidth=2, color='blue')
plt.plot(res_df['SMOTE Ratio'], res_df['F1-Score'], marker='^', label='Fraud F1-Score', linewidth=2, color='green', linestyle='--')

plt.title('Impact of SMOTE Balancing Ratio on Fraud Detection Performance', fontsize=14)
plt.xlabel('Oversampling Ratio (Minority as % of Majority)', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Figure_SMOTE_Impact_Study.png', dpi=300, bbox_inches='tight')
plt.show()


print("\n✅ Sensitivity Analysis Complete. Results table and Figure_SMOTE_Impact_Study.png generated.")
