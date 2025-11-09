Anomaly-Based IDS using Random Forest on CICIOT23

This project implements a complete, end-to-end Anomaly-Based Intrusion Detection System (AIDS) using a Random Forest classifier. The model is trained on the CICIOT23 dataset to distinguish between benign network traffic and various types of cyber-attacks.

Tech Stack

Python

scikit-learn: For the Random Forest model and metrics

pandas: For data manipulation

matplotlib & seaborn: For plotting

Kaggle: The project is designed to run in a Kaggle Notebook environment.

Automated ML Pipeline

The script automates the entire machine learning pipeline, from raw data to a deployable model and final reports.

graph TD
    A[Start: Raw CICIOT23 Data] --> B{Phase 0: Merge CSVs};
    B --> C[merged_train.csv];
    B --> D[merged_test.csv];
    B --> E[merged_validation.csv];
    
    C --> F{Phase 1: Model Training};
    F --> G[ids_scaler.pkl];
    F --> H[rf_ids_model.pkl];
    
    D --> I{Phase 2: Model Evaluation};
    G --> I;
    H --> I;
    I --> J[Accuracy & Metrics];
    I --> K[Confusion Matrix];
    I --> L[Feature Importance Plot];
    
    E --> M{Phase 3: Live Detection};
    G --> M;
    H --> M;
    M --> N[anomalous_flows_report.html];
    M --> O[anomalous_flows_report.csv];


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

Run the cell. The entire pipeline will execute.

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

(Note: Assumes you have created an images folder in your repository)

Feature Importance

The model identified Inter-Arrival Time (IAT) and specific flag counts (rst_count, urg_count) as the most significant indicators of an anomaly.

(Note: Assumes you have created an images folder in your repository)
