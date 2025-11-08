# CICIOT23-RandomForest-IDS
An Anomaly-Based Intrusion Detection System (AIDS) built with a Random Forest classifier on the CICIOT23 dataset. This project automates the full ML pipeline to detect anomalous IoT network traffic with 99.76% accuracy.
Anomaly-Based IDS using Random Forest on CICIOT23.

This project implements a complete, end-to-end Anomaly-Based Intrusion Detection System (AIDS) using a Random Forest classifier. The model is trained on the CICIOT23 dataset to distinguish between benign network traffic and various types of cyber-attacks.

Tech Stack

Python

scikit-learn: For the Random Forest model and metrics

pandas: For data manipulation

matplotlib & seaborn: For plotting

Kaggle: The project is designed to run in a Kaggle Notebook environment.

How It Works

The script automates the entire machine learning pipeline in four phases:

Phase 0: Data Merging

Automatically finds all .csv files in the train/, test/, and validation/ directories.

Merges them into three single files (merged_train.csv, merged_test.csv, merged_validation.csv) for efficient processing.

Phase 1: Model Training

Loads and preprocesses the merged_train.csv file.

Converts labels to binary: BenignTraffic -> 0, all other labels -> 1 (Anomaly).

Scales features using StandardScaler.

Trains a RandomForestClassifier on the scaled data.

Saves the trained model (rf_ids_model.pkl) and scaler (ids_scaler.pkl).

Phase 2: Model Evaluation

Loads the merged_test.csv file (unseen data).

Evaluates the trained model's performance.

Generates key metrics, a classification report, a confusion matrix, and a feature importance plot.

Phase 3: "Live" Detection Simulation

Loads the saved rf_ids_model.pkl and ids_scaler.pkl.

Loads the merged_validation.csv file to simulate new, "live" network traffic.

Detects and flags all anomalous flows.

Generates anomalous_flows_report.csv and anomalous_flows_report.html as the final output.

How to Run

Get the Data: Download the CICIOT23 dataset from Kaggle.

Set up Environment:

Create a new Kaggle Notebook and add the CICIOT23 dataset as input.

Ensure your dataset path in the notebook is /kaggle/input/ciciot2023/.

If your path is different, update the BASE_DATA_PATH variable in kaggle_aids_rf.py.

Install Libraries:

pip install -r requirements.txt


Run the Script:

Add the kaggle_aids_rf.py script to a cell in your notebook.

Run the cell. The entire pipeline (Phases 0-3) will execute.

All outputs (model, scaler, reports) will be saved to /kaggle/working/.

Final Results

The model performs with extremely high accuracy, demonstrating its effectiveness in identifying anomalous traffic.

Performance Metrics

Overall Accuracy: 99.76%

Anomaly Precision: 99.90%

Anomaly Recall: 99.85%

Anomaly F1-Score: 0.9988

Classification Report

              precision    recall  f1-score   support

  Benign (0)       0.94      0.96      0.95     27709
 Anomaly (1)       1.00      1.00      1.00   1149142

    accuracy                           1.00   1176851
   macro avg       0.97      0.98      0.97   1176851
weighted avg       1.00      1.00      1.00   1176851


Confusion Matrix

The confusion matrix shows a very low number of False Positives (1,123) and False Negatives (1,711) compared to the millions of correct predictions.

Feature Importance

The model identified Inter-Arrival Time (IAT) and specific flag counts (rst_count, urg_count) as the most significant indicators of an anomaly.
