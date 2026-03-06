# @title Automated 5-Fold SMOTE Ratio Impact Study
# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
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
# 1. LOADING & DATA PREPARATION
# ==============================================================================
inputFile = 'preprocessed_data.csv'
text_column = 'message'
label_column = 'label'
RANDOM_STATE = 42
NUMERICAL_FEATURES = ['uppercase_ratio', 'symbol_count', 'word_count', 
                      'url_count', 'phone_token_count', 'currency_token_count']

df = pd.read_csv(inputFile)
for col in NUMERICAL_FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
df.dropna(subset=[text_column, label_column], inplace=True)

X_text = df[text_column]
X_num = df[NUMERICAL_FEATURES]
y = df[label_column].values

# ==============================================================================
# 2. 5-FOLD CROSS-VALIDATION WITH RATIO LOOP
# ==============================================================================
ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
cv_results = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print(f"--- Running 5-Fold Cross-Validation Study ---")
print(f"{'Ratio':<10} | {'Avg Recall':<10} | {'Avg Precision':<13} | {'Avg F1':<8}")
print("-" * 55)

for ratio in ratios:
    fold_metrics = []
    
    for train_index, test_index in skf.split(X_text, y):
        # Split Data
        X_t_train, X_t_test = X_text.iloc[train_index], X_text.iloc[test_index]
        X_n_train, X_n_test = X_num.iloc[train_index], X_num.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Vectorization (Fit ONLY on training fold)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
        X_tfidf_train = vectorizer.fit_transform(X_t_train)
        X_tfidf_test = vectorizer.transform(X_t_test)

        # Combine Features
        X_train_comb = hstack([X_tfidf_train, X_n_train.values]).tocsr()
        X_test_comb = hstack([X_tfidf_test, X_n_test.values]).tocsr()

        # Apply SMOTE ONLY to training data
        if ratio > 0:
            sm = SMOTE(sampling_strategy=ratio, random_state=RANDOM_STATE)
            X_train_res, y_train_res = sm.fit_resample(X_train_comb, y_train)
        else:
            X_train_res, y_train_res = X_train_comb, y_train

        # Train & Predict
        clf = lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
        clf.fit(X_train_res, y_train_res)
        preds = clf.predict(X_test_comb)

        # Metrics for Fraud Class
        fold_metrics.append([
            recall_score(y_test, preds),
            precision_score(y_test, preds),
            f1_score(y_test, preds)
        ])

    # Average metrics across 5 folds
    avg_metrics = np.mean(fold_metrics, axis=0)
    label_text = "No SMOTE" if ratio == 0 else f"{int(ratio*100)}%"
    
    cv_results.append({
        'Ratio': label_text,
        'Recall': avg_metrics[0],
        'Precision': avg_metrics[1],
        'F1-Score': avg_metrics[2]
    })
    
    print(f"{label_text:<10} | {avg_metrics[0]:<10.4f} | {avg_metrics[1]:<13.4f} | {avg_metrics[2]:<8.4f}")

# ==============================================================================
# 3. VISUALIZATION
# ==============================================================================
res_df = pd.DataFrame(cv_results)
plt.figure(figsize=(12, 6))
plt.plot(res_df['Ratio'], res_df['Recall'], marker='o', label='Avg Recall', color='red', lw=2)
plt.plot(res_df['Ratio'], res_df['Precision'], marker='s', label='Avg Precision', color='blue', lw=2)
plt.plot(res_df['Ratio'], res_df['F1-Score'], marker='^', label='Avg F1-Score', color='green', lw=2, ls='--')

plt.title('5-Fold Cross-Validation: Impact of SMOTE Ratios on Fraud Detection', fontsize=14)
plt.xlabel('Oversampling Ratio (Minority as % of Majority)', fontsize=12)
plt.ylabel('Average Score (5 Folds)', fontsize=12)
plt.legend()
plt.savefig('Figure_5Fold_SMOTE_Study.png', dpi=300)
plt.show()