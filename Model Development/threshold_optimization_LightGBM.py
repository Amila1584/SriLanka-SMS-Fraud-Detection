# @title
# ==============================================================================
# 0. IMPORTS (Needed for thresholding)
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# NOTE: Assuming LGBMClassifier, X_test_combined, y_test, and best_model are available from Section 6

# ==============================================================================
# 7. STEP 7: CLASSIFICATION THRESHOLD OPTIMIZATION
# ==============================================================================
print("\n--- [7] CLASSIFICATION THRESHOLD OPTIMIZATION ---")

# Define file names for this section
PLOT_FILENAME_THRESHOLDS = 'Figure_7_1_Threshold_Optimization.png'
PLOT_FILENAME_CM_FINAL = 'Figure_7_2_CM_LGBM_Final_Tuned.png'

# 7.1 Generate Probabilities
try:
    # Use the best model (Tuned LightGBM) to predict probabilities for the positive class (Fraud)
    # NOTE: Assuming 'best_model' and 'X_test_combined' are loaded from the previous step.
    y_prob = best_model.predict_proba(X_test_combined)[:, 1]
except NameError:
    # Handle the case if the notebook was reset and best_model/X_test_combined aren't found
    print("❌ ERROR: Required variables (best_model, X_test_combined) not found. Please ensure Section 6 ran successfully.")
    exit()

# 7.2 Define Optimization Parameters
thresholds = np.arange(0.05, 0.96, 0.01)
scores = []

# 7.3 Iterate and Calculate Metrics (F1-Score Focus)
for t in thresholds:
    y_pred_thresh = (y_prob >= t).astype(int)
    # Calculate scores, handling potential zero division gracefully
    rec = recall_score(y_test, y_pred_thresh, pos_label=1)
    prec = precision_score(y_test, y_pred_thresh, pos_label=1) if np.sum(y_pred_thresh) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    scores.append({
        'Threshold': t,
        'F1_Score': f1,
        'Recall': rec,
        'Precision': prec
    })

results_df = pd.DataFrame(scores)

# Find the optimal threshold that maximizes the F1-Score
optimal_threshold_row = results_df.loc[results_df['F1_Score'].idxmax()]
optimal_threshold = optimal_threshold_row['Threshold']
print(f"✅ Optimal Threshold (Max F1-Score): {optimal_threshold:.2f}")

# 7.4 Evaluate Model with Optimal Threshold
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
mcc_optimal = matthews_corrcoef(y_test, y_pred_optimal)

# 7.5 Print Classification Report and Confusion Matrix (Text Output)
print("\n## 7.5 CLASSIFICATION REPORT (Optimal Threshold)")
print("---------------------------------------------------------")
print(classification_report(y_test, y_pred_optimal, target_names=['Legitimate (0)', 'Fraud (1)']))

print("\n## 7.5 CONFUSION MATRIX (Optimal Threshold)")
print("------------------------------------------")
cm_text = pd.DataFrame(cm_optimal, index=['Actual Legit (0)', 'Actual Fraud (1)'], columns=['Predicted Legit (0)', 'Predicted Fraud (1)'])
print(cm_text.to_string())

# 7.6 Threshold Visualization (Figure 7.1)
plt.figure(figsize=(10, 7))
plt.plot(results_df['Threshold'], results_df['F1_Score'], label='F1-Score (Optimization Target)', color='green', linewidth=2)
plt.plot(results_df['Threshold'], results_df['Recall'], label='Recall', color='red', linestyle='--')
plt.plot(results_df['Threshold'], results_df['Precision'], label='Precision', color='blue', linestyle='--')
plt.axvline(optimal_threshold, color='black', linestyle=':', label=f'Optimal F1 Threshold: {optimal_threshold:.2f}')

plt.xlabel('Classification Threshold', fontsize=12)
plt.ylabel('Performance Metric Value', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--')
plt.savefig(PLOT_FILENAME_THRESHOLDS, dpi=300, bbox_inches='tight')
print(f"✅ Figure 7.1 (Threshold Optimization Curve) saved as: {PLOT_FILENAME_THRESHOLDS}")
plt.show()

# 7.7 Final Reporting and CM Visualization
report_optimal = classification_report(y_test, y_pred_optimal, output_dict=True)

print("\n## 7.7 FINAL PERFORMANCE SUMMARY")
print("---------------------------------")
print(f"**Optimal Threshold Used:** {optimal_threshold:.2f}")
print(f"**Matthews Correlation Coefficient (MCC):** {mcc_optimal:.4f}")
print(f"**Fraud Recall (Detection Rate):** {report_optimal['1']['recall']:.4f}")
print(f"**Fraud Precision (Usability):** {report_optimal['1']['precision']:.4f}")


# Display Optimal Confusion Matrix (Figure 7.2)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Legit', 'Predicted Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud'])
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.savefig(PLOT_FILENAME_CM_FINAL, dpi=300, bbox_inches='tight')
print(f"✅ Final CM saved as: {PLOT_FILENAME_CM_FINAL}")
plt.show()

print("\n--- Saved Plot Filenames ---")
print(f"✅ Visualization 1 - Threshold Optimization saved as: {PLOT_FILENAME_THRESHOLDS}")
print(f"✅ Visualization 2 - Final Confusion Matrix saved as: {PLOT_FILENAME_CM_FINAL}")