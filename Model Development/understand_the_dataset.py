# @title
# ==============================================================================
# 0. IMPORTS
# ==============================================================================
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns

# Install tabulate for markdown output in pandas
!pip install tabulate

# Configure settings for high-quality plots for dissertation output
sns.set_theme(style="whitegrid")
# Use a non-interactive backend for saving figures reliably
plt.switch_backend('Agg')

# ==============================================================================
# 1. CONFIGURATION: Defining Research Parameters
# ==============================================================================
print("--- [1] CONFIGURATION AND SETUP ---")
inputFile = 'final_dataset_v4.csv'
text_column = 'message'
label_column = 'label'
FRAUD_LABEL = 1 # Defining 'Fraud' as the positive class for Recall optimization
LEGIT_LABEL = 0 # Defining 'Legit' as the negative class
SAMPLE_SIZE = 5 # Number of random samples for visual inspection
RANDOM_STATE = 42 # Ensuring reproducibility for all random sampling

# Define output file names for high-resolution visualizations
PLOT_FILENAME_IMBALANCE = 'Figure_1_1_Class_Imbalance_Log.png'
PLOT_FILENAME_LENGTH = 'Figure_1_2_Message_Length_Distribution.png'


# ==============================================================================
# 2. STEP 2: DATA ACQUISITION AND INITIAL PRE-PROCESSING
# ==============================================================================
print("\n--- [2] DATA LOADING AND PRE-PROCESSING ---")
df = None
try:
    # 2.1 Data Loading: Using the robust Python engine to handle potential formatting errors
    df = pd.read_csv(
        inputFile,
        encoding='utf-8',
        on_bad_lines='skip',
        engine='python',
        header=None # Assuming no formal header to control column assignment manually
    )

    initial_rows = len(df)
    print(f"✅ Dataset '{inputFile}' loaded successfully.")
    print(f"   Initial total data points loaded: {initial_rows}")

    # 2.2 CRITICAL FIX: Explicit Column Assignment
    if len(df.columns) < 2:
        raise ValueError("The dataset must contain at least two columns: message and label.")

    # FIX: Assigning columns as [message, label, ...] based on the assumed file structure.
    df.columns = [text_column, label_column] + list(df.columns[2:])
    print(f"   Columns explicitly set to: [{text_column}, {label_column}, ...]")

    # 2.3 Data Type Enforcement and Integrity Check

    # Coerce the label column to numeric
    df[label_column] = pd.to_numeric(df[label_column], errors='coerce')

    rows_before_drop = len(df)
    # Removing any row where the core data (message or label) is corrupt (NaN)
    df.dropna(subset=[label_column, text_column], inplace=True)

    # Final enforcement of the target variable type
    df[label_column] = df[label_column].astype(int)

    # Message Content Cleaning: Stripping whitespace and converting to lowercase
    df[text_column] = df[text_column].astype(str).str.strip().str.lower()
    df.drop(df[df[text_column].str.len() == 0].index, inplace=True) # Remove empty messages

    rows_after_drop = len(df)
    print(f"   Corrupt or invalid rows removed during cleansing: {rows_before_drop - rows_after_drop}")
    print(f"   Final valid data points ready for analysis: {rows_after_drop}")


except Exception as e:
    print(f"❌ A critical error occurred during data processing: {e}")
    df = None

# Proceed only if DataFrame is valid
if df is not None and len(df) > 0:
    # ==============================================================================
    # 3. STEP 3: ACADEMIC DATA EXPLORATION AND CLASS ANALYSIS
    # ==============================================================================
    print("\n--- [3] ACADEMIC DATA EXPLORATION AND CLASS ANALYSIS ---")

    print("## 3.1 Representative Sample Data Preview (Randomly Selected)")
    print("------------------------------------------------------------")

    # Separating classes for detailed analysis
    df_legit = df[df[label_column] == LEGIT_LABEL]
    df_fraud = df[df[label_column] == FRAUD_LABEL]

    # Sampling for visual confirmation (using the defined random state for reproducibility)
    legit_samples = df_legit[[text_column, label_column]].sample(n=min(SAMPLE_SIZE, len(df_legit)), random_state=RANDOM_STATE)
    fraud_samples = df_fraud[[text_column, label_column]].sample(n=min(SAMPLE_SIZE, len(df_fraud)), random_state=RANDOM_STATE)

    print(f"\n### 3.1.1 Random {min(SAMPLE_SIZE, len(df_legit))} Legitimate Samples (Label: {LEGIT_LABEL})")
    print("--------------------------------------------------")
    print(legit_samples.to_markdown(index=False))

    print(f"\n### 3.1.2 Random {min(SAMPLE_SIZE, len(df_fraud))} Fraudulent Samples (Label: {FRAUD_LABEL})")
    print("--------------------------------------------------")
    print(fraud_samples.to_markdown(index=False))

    print("\n## 3.2 Dataset Integrity and Types")
    print("----------------------------------")
    df.info()

    print("\n## 3.3 Critical Class Distribution Analysis")
    print("--------------------------------------------")

    # Quantifying the class imbalance
    class_counts = df[label_column].value_counts(dropna=False)
    total_samples = len(df)

    if (FRAUD_LABEL in class_counts.index and
        LEGIT_LABEL in class_counts.index):

        fraud_count = class_counts.get(FRAUD_LABEL, 0)
        legit_count = class_counts.get(LEGIT_LABEL, 0)

        fraud_percentage = (fraud_count / total_samples) * 100

        # Presentation
        data_summary = {
            'Class Label': ['Legitimate (0)', 'Fraudulent (1)', 'Total Samples'],
            'Count': [legit_count, fraud_count, total_samples],
            'Proportion (%)': [f"{legit_count / total_samples * 100:.4f}", f"{fraud_percentage:.4f}", "100.0000"]
        }

        distribution_df = pd.DataFrame(data_summary)
        print(distribution_df.to_markdown(index=False))

        # Explicitly addressing the imbalance
        imbalance_ratio = legit_count / fraud_count if fraud_count > 0 else np.inf

        print(f"\n🔑 Key Finding: Severe Class Imbalance (Ratio: {imbalance_ratio:.2f}:1)")
        print("----------------------------------------------------------------------")
        print(f"The fraudulent class (our **positive case**) constitutes only **{fraud_percentage:.4f}%** of the dataset.")
        print("\n**Methodological Implication:**")
        print("This extreme imbalance confirms that simply optimizing for **Accuracy** is misleading. Our strategy must focus on specialized techniques (like cost-sensitive learning) to maximize **Recall** and ensure true fraud detection.")

    else:
        print("Error: Expected class labels (0 and 1) were not found. Cannot perform full class analysis.")

    # ==============================================================================
    # 4. STEP 4: PRE-PROCESSING FOR VISUALIZATION
    # ==============================================================================
    print("\n--- [4] PRE-PROCESSING FOR VISUALIZATION ---")

    # Calculate message length (in characters) as a key feature
    df['message_length'] = df[text_column].apply(len)
    print("✅ Message length (character count) calculated for visualization analysis.")


    # ==============================================================================
    # 5. STEP 5: ADVANCED DATA VISUALIZATIONS (SAVE and DISPLAY)
    # ==============================================================================
    print("\n--- [5] GENERATING ADVANCED DATA VISUALIZATIONS (Thesis Ready) ---")

    ## 5.1 Visualization 1: Class Imbalance Confirmation (Log Scale Bar Plot)
    try:
        fig1, ax1 = plt.subplots(figsize=(8, 6))

        # Plotting the counts for visual confirmation of the massive skew
        sns.barplot(
            x=['Legitimate (0)', 'Fraudulent (1)'],
            y=[legit_count, fraud_count],
            palette=['#4daf4a', '#e41a1c'], # Consistent color scheme (Green for Legit, Red for Fraud)
            ax=ax1
        )
        #ax1.set_title('Class Distribution', fontsize=14, weight='bold')
        ax1.set_ylabel('Number of SMS Messages (Log Scale)', fontsize=12)
        ax1.set_xlabel('Class Label', fontsize=12)

        # CRUCIAL: Using LOG SCALE on the Y-axis to visualize the fraudulent minority
        ax1.set_yscale('log')

        # Adding labels for the majority class (legitimate)
        ax1.text(0, legit_count, f'{legit_count:,}', ha='center', va='bottom', fontsize=10)

        # ***FINAL CRITICAL FIX IMPLEMENTED: Repositioning and styling the Fraudulent Count Label***
        # Y-position manually adjusted to 1800 (from 2000) on the log scale to sit closer to the bar (1581).
        # Color changed to black, and weight removed.
        ax1.text(1, 1800, f'{fraud_count:,}', ha='center', va='bottom', fontsize=10, color='black')


        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the high-resolution figure for the thesis
        fig1.savefig(PLOT_FILENAME_IMBALANCE, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization 1 (Class Imbalance) saved as: {PLOT_FILENAME_IMBALANCE}")

        # Display the figure in the cell output
        plt.show()

    except Exception as e:
        print(f"❌ Error generating Imbalance Visualization: {e}")

    ## 5.2 Visualization 2: Message Length Distribution (Violin Plot)
    try:
        fig2, ax2 = plt.subplots(figsize=(10, 7))

        # Violin plot shows the density and quartile distribution for a key feature (length)
        sns.violinplot(
            x=label_column,
            y='message_length',
            data=df,
            palette=['#4daf4a', '#e41a1c'],
            split=False,
            inner='box', # Shows min/max/median/quartiles inside the violin
            ax=ax2
        )

        #x2.set_title('Message Length Distribution by Class', fontsize=14, weight='bold')
        ax2.set_xlabel('Class Label (0: Legitimate, 1: Fraudulent)', fontsize=12)
        ax2.set_ylabel('Message Length (Characters)', fontsize=12)

        # Focus the Y-axis on the central distribution (99th percentile)
        ax2.set_ylim(0, df['message_length'].quantile(0.99) + 10)

        # Setting clear labels
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Legitimate Messages', 'Fraudulent Messages'])

        # Save the high-resolution figure for the thesis
        fig2.savefig(PLOT_FILENAME_LENGTH, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization 2 (Message Length) saved as: {PLOT_FILENAME_LENGTH}")

        # Display the figure in the cell output
        plt.show()

    except Exception as e:
        print(f"❌ Error generating Length Distribution Visualization: {e}")
