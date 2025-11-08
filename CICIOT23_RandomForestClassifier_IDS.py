import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
import glob

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---

# !!! IMPORTANT !!!
# Update this path to match your Kaggle dataset's input directory.
BASE_DATA_PATH = "/kaggle/input/ciciot2023/CICIOT23/"

# Source directories for merging
SOURCE_TRAIN_DIR = os.path.join(BASE_DATA_PATH, "train")
SOURCE_TEST_DIR = os.path.join(BASE_DATA_PATH, "test")
SOURCE_VALIDATE_DIR = os.path.join(BASE_DATA_PATH, "validation")

# Directory to save outputs (model, scaler, reports)
OUTPUT_DIR = "/kaggle/working/"

# These paths now point to /kaggle/working/ where we will save the merged files
TRAIN_FILE_PATH = os.path.join(OUTPUT_DIR, "merged_train.csv")
TEST_FILE_PATH = os.path.join(OUTPUT_DIR, "merged_test.csv")
VALIDATE_FILE_PATH = os.path.join(OUTPUT_DIR, "merged_validation.csv")


# !!! IMPORTANT !!!
# Update this to the name of your target/label column in the CSV.
TARGET_COLUMN = "label"

# !!! IMPORTANT !!!
# Update this to the string that represents a "normal" or "benign" connection.
# All other labels will be treated as anomalies (1).
BENIGN_LABEL_VALUE = "BenignTraffic"

# Output paths for model and reports
MODEL_PATH = os.path.join(OUTPUT_DIR, "rf_ids_model.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "ids_scaler.pkl")
HTML_REPORT_PATH = os.path.join(OUTPUT_DIR, "anomalous_flows_report.html")
CSV_REPORT_PATH = os.path.join(OUTPUT_DIR, "anomalous_flows_report.csv")

# --- 1. Data Merging ---

def merge_csvs(folder_path, output_file):
    """
    Globs all CSVs in a folder, merges them, and saves to a new file.
    Returns True on success, False on failure.
    """
    print(f"Searching for CSVs in: {folder_path}")
    files = glob.glob(f"{folder_path}/*.csv")
    if not files:
        print(f"--- ERROR: No CSV files found in {folder_path} ---")
        print("Please check your `BASE_DATA_PATH` and directory names.")
        return False
        
    print(f"Found {len(files)} CSVs. Merging...")
    df_list = []
    for f in files:
        try:
            df_list.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"Could not read {f}: {e}")
    
    if not df_list:
        print(f"--- ERROR: No dataframes to merge for {folder_path} ---")
        return False
        
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"Saving merged file to: {output_file}")
    merged_df.to_csv(output_file, index=False)
    print(f"Save complete. Total rows: {len(merged_df)}")
    return True

# --- 2. Data Loading and Preprocessing ---

def load_and_preprocess(file_path):
    """
    Loads data, handles non-numeric values, and converts to binary labels.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"--- ERROR: File not found at {file_path} ---")
        print("Please check your `BASE_DATA_PATH` and file name configurations.")
        return None, None, None

    # Handle potential infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    if TARGET_COLUMN not in df.columns:
        print(f"--- ERROR: Target column '{TARGET_COLUMN}' not found in data. ---")
        return None, None, None

    print("Data loaded. Preprocessing labels...")
    # Convert labels to binary: 0 for Benign, 1 for Anomaly
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
        lambda x: 0 if x == BENIGN_LABEL_VALUE else 1
    )
    
    y = df[TARGET_COLUMN]
    X = df.drop(TARGET_COLUMN, axis=1)
    
    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    # Drop any columns that became all-NaN after coercion
    X.dropna(axis=1, how='all', inplace=True)
    # Re-align X and y
    y = y[X.index]

    print(f"Preprocessing complete. Features: {X.shape[1]}, Samples: {X.shape[0]}")
    return X, y, X.columns.tolist()

# --- 3. Model Training and Scaling ---

def train_model(X_train, y_train):
    """
    Scales features and trains a Random Forest classifier.
    """
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Training Random Forest model...")
    # n_jobs=-1 uses all available cores
    # random_state=42 ensures reproducible results
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    print("Model training complete.")
    
    print(f"Saving scaler to {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saving model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    
    return model, scaler, X_train_scaled

# --- 4. Model Evaluation ---

def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluates the model on the test set and prints/plots results.
    """
    print("Evaluating model on test set...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (Anomaly)")
    print(f"Recall:    {recall:.4f} (Anomaly)")
    print(f"F1-Score:  {f1:.4f} (Anomaly)")
    
    print("\nClassification Report:")
    # Force report to include both classes
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=["Benign (0)", "Anomaly (1)"],
        labels=[0, 1]
    )
    print(report)
    
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)
    
# --- 5. Visualization (Graphs) ---

def plot_confusion_matrix(cm):
    """
    Plots a heatmap for the confusion matrix.
    """
    print("Plotting confusion matrix...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        xticklabels=["Predicted Benign", "Predicted Anomaly"],
        yticklabels=["Actual Benign", "Actual Anomaly"]
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plots the top 20 most important features.
    """
    print("Plotting feature importance...")
    importances = model.feature_importances_
    
    print("--- Top 5 Feature Importances ---")
    indices_top5 = np.argsort(importances)[-5:]
    for i in reversed(indices_top5): # Show in descending order
        print(f"{feature_names[i]:<30}: {importances[i]:.6f}")
    print("---------------------------------")
    
    indices = np.argsort(importances)[-20:]
    
    # Filter to only plot features with importance > 0
    plot_indices = [i for i in indices if importances[i] > 0]
    plot_importances = importances[plot_indices]
    plot_features = [feature_names[i] for i in plot_indices]
    
    if not plot_features:
        print("No features with importance > 0.0 found. Skipping plot.")
        return

    plt.figure(figsize=(12, 8))
    plt.title("Top Feature Importances (showing > 0.0)")
    plt.barh(range(len(plot_indices)), plot_importances, color="b", align="center")
    plt.yticks(range(len(plot_indices)), plot_features)
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.show()

# --- 6. "Live" Detection and Reporting ---

def detect_anomalies_from_csv(input_csv_path, model_path, scaler_path, train_features):
    """
    Loads the saved model and scaler to detect anomalies in a new CSV file.
    Generates HTML and CSV reports of detected anomalies.
    """
    print("\n--- Simulating 'Live' Detection ---")
    print(f"Loading model from {model_path}")
    print(f"Loading scaler from {scaler_path}")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print("--- ERROR: Model or scaler file not found. ---")
        print("Please run the training process first.")
        return

    print(f"Loading 'live' data from {input_csv_path}...")
    try:
        live_df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"--- ERROR: Live data file not found at {input_csv_path} ---")
        return

    original_df = live_df.copy()

    if TARGET_COLUMN in live_df.columns:
        live_features = live_df.drop(TARGET_COLUMN, axis=1)
    else:
        live_features = live_df
        
    live_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Align columns to match training data
    if train_features is None:
        print("--- ERROR: Could not get training feature list for alignment. ---")
        return
        
    live_features = live_features.reindex(columns=train_features, fill_value=0)
    live_features.fillna(0, inplace=True)

    print("Scaling 'live' features...")
    live_features_scaled = scaler.transform(live_features)
    
    print("Predicting anomalies...")
    predictions = model.predict(live_features_scaled)
    
    original_df['detection_result'] = predictions
    original_df['detection_label'] = original_df['detection_result'].apply(
        lambda x: "Anomaly" if x == 1 else "Benign"
    )
    
    anomalous_flows = original_df[original_df['detection_result'] == 1]
    
    if anomalous_flows.empty:
        print("No anomalies detected in the 'live' data.")
    else:
        print(f"Detected {len(anomalous_flows)} anomalous flows.")
        
        print(f"Saving CSV report to {CSV_REPORT_PATH}")
        anomalous_flows.to_csv(CSV_REPORT_PATH, index=False)
        
        print(f"Saving HTML report to {HTML_REPORT_PATH}")
        anomalous_flows.to_html(HTML_REPORT_PATH, index=False, border=1)
        
    print("--- 'Live' Detection Simulation Complete ---")

# --- Main Execution ---

def main():
    print("--- Project 1: Anomaly-Based IDS using Random Forest ---")
    
    # --- Phase 0: Merging CSV Files ---
    print("\n--- Phase 0: Merging CSV Files ---")
    
    if not merge_csvs(SOURCE_TRAIN_DIR, TRAIN_FILE_PATH):
        print("Halting: Could not merge training data.")
        return
        
    if not merge_csvs(SOURCE_TEST_DIR, TEST_FILE_PATH):
        print("Halting: Could not merge test data.")
        return
        
    if not merge_csvs(SOURCE_VALIDATE_DIR, VALIDATE_FILE_PATH):
        print("Halting: Could not merge validation data.")
        return

    # --- Phase 1: Training Phase ---
    print("\n--- Phase 1: Training Phase ---")
    X_train, y_train, feature_names = load_and_preprocess(TRAIN_FILE_PATH)
    X_test, y_test, _ = load_and_preprocess(TEST_FILE_PATH)
    
    if X_train is None or X_test is None:
        print("Halting execution due to data loading errors.")
        return
        
    model, scaler, X_train_scaled = train_model(X_train, y_train)
    
    # --- Phase 2: Evaluation Phase ---
    print("\n--- Phase 2: Evaluation Phase ---")
    evaluate_model(model, scaler, X_test, y_test)
    plot_feature_importance(model, feature_names)

    # --- Phase 3: Detection Phase ---
    print("\n--- Phase 3: Detection Phase ---")
    detect_anomalies_from_csv(VALIDATE_FILE_PATH, MODEL_PATH, SCALER_PATH, feature_names)
    
    print("\n--- Script Finished ---")
    print(f"All outputs (model, scaler, reports) are saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()