Getting Started
Clone the repository:

Bash
git clone [https://github.com/yourusername/predictive-maintenance.git](https://github.com/yourusername/predictive-maintenance.git)
cd predictive-maintenance
Data Placement:
Ensure your predictive_maintenance_dataset.csv is located in your desired directory and update the path in the script accordingly.

Run the Pipeline:
Execute the main python script to perform feature engineering, train the model, and generate evaluation visualisations:

Bash
python "predictive maintainence.py"
📊 Output Visualizations
The script automatically generates several advanced visualizations to evaluate model performance:

Confusion Matrix: Evaluates the predictions using the optimal threshold, showing true positives (caught failures) against false positives (false alarms).

Feature Importance Chart: Isolates the top 20 engineered features driving the Random Forest's decision-making.

Precision-Recall Curve (PR-AUC): Because the dataset is highly imbalanced, ROC-AUC can be misleading. The PR Curve highlights the trade-off between True Alarm Rate and Caught Failures, pinpointing the optimal threshold coordinate.

Created as part of a Data Science & Machine Learning Portfolio.
"""

with open('Predictive_Maintenance_Github_Readme.md', 'w', encoding='utf-8') as f:
f.write(readme_content)

print("Predictive_Maintenance_Github_Readme.md generated")

```python?code_reference&code_event_index=6
readme_content = """# Predictive Maintenance: Overcoming Extreme Class Imbalance

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Engineering-yellow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-red)

## 📌 Project Overview

This repository contains an end-to-end machine learning pipeline designed to predict equipment failures using daily telemetry data. 

In predictive maintenance, the most significant challenge is **extreme class imbalance**. Because machines operate normally the vast majority of the time, failure events are incredibly rare (often <0.1% of records). Standard algorithms trained on this data will simply learn to predict that "nothing will ever fail" and still achieve >99.8% accuracy (the Accuracy Paradox).

This project resolves this challenge through two primary techniques:
1. **Time-Series Feature Engineering:** Creating rolling windows and velocity metrics to catch degradation trends before a failure occurs.
2. **SMOTE (Synthetic Minority Over-sampling Technique):** Synthetically generating minority class examples during training to force the model to learn the patterns of failure.

---

## 📑 Table of Contents
1. [Dataset Overview](#-dataset-overview)
2. [Methodology & Architecture](#-methodology--architecture)
3. [Installation & Setup](#%EF%B8%8F-installation--setup)
4. [Feature Engineering Strategy](#-feature-engineering-strategy)
5. [Model Evaluation & Threshold Optimization](#-model-evaluation--threshold-optimization)
6. [Key Takeaways](#-key-takeaways)

---

## 🗄️ Dataset Overview

The dataset consists of daily telemetry logs collected from various devices over time.

* **Total Records:** 124,494 daily logs
* **Unique Devices:** 1,169
* **Target Variable:** `failure` (0 = Healthy, 1 = Failure)
* **Class Distribution:** 99.91% Healthy vs. 0.08% Failure (Extreme Imbalance)
* **Raw Features:** Date, Device ID, and 9 numeric telemetry metrics (`metric1` - `metric9`).

---

## 🔬 Methodology & Architecture

1. **Chronological Sorting:** The dataset is strictly sorted by `device` and `date` to ensure that time-series functions (like `.rolling()` or `.diff()`) do not leak data across different devices or timeframes.
2. **Feature Expansion:** The 8 core metrics are expanded into a comprehensive 48-feature space using historical calculations.
3. **Train/Test Split:** Data is split 80/20 using stratified sampling to ensure both sets contain a proportional number of the rare failure events.
4. **Data Balancing (SMOTE):** To prevent the model from ignoring the minority class, SMOTE is applied **only to the training set** to balance the classes. *Applying it before the split would result in critical data leakage.*
5. **Model Training:** A `RandomForestClassifier` is trained on the resampled, balanced data.
6. **Probability & Thresholding:** Instead of relying on the default 0.5 classification threshold, the optimal threshold is calculated dynamically using the Precision-Recall Curve to maximize the F1-Score.

---

## ⚙️ Installation & Setup

### Prerequisites
Make sure you have Python installed along with the following required libraries:

Running the Pipeline
Clone this repository.

Ensure you have the predictive_maintenance_dataset.csv in your working directory (or update the file path in the code).

Execute the script:

Bash
python "predictive maintainence.py"
⏱️ Feature Engineering Strategy
Raw telemetry data at a single point in time is rarely indicative of a mechanical failure. Predictive maintenance requires understanding how the machine's behavior is changing. The script automatically generates the following features for each metric:

Rolling Averages: 3-day and 7-day smoothing windows to reduce noise (rolling_3d, rolling_7d).

Velocity Metrics: 1-day absolute changes (change_1d) and percentage changes (change_pct_1d) to catch sudden spikes or drops.

Trend Indicators: A binary feature calculating whether the short-term rolling average is currently moving upward or downward (trend_3d).

📊 Model Evaluation & Threshold Optimization
Because the dataset is heavily imbalanced, overall "Accuracy" and the standard ROC-AUC metrics can be highly misleading.

Output Metrics Evaluated:
Precision-Recall Curve (PR-AUC): The script plots the PR Curve, which is the gold standard for evaluating highly imbalanced datasets.

Optimal Threshold Calculation: The script calculates the exact F1-Score at every possible probability threshold and selects the point that perfectly balances Precision (minimizing false alarms) and Recall (catching actual failures).

Confusion Matrix: Evaluates the final predictions using this optimized threshold to give a realistic view of how the model would perform in a production environment.

Feature Importance: Identifies and plots the top 20 engineered time-series features that the Random Forest algorithm relied upon the most to predict failure.

Created as part of an Advanced Data Science Portfolio.
"""

with open('Predictive_Maintenance_Github_README.md', 'w', encoding='utf-8') as f:
f.write(readme_content)

print("Predictive_Maintenance_Github_README.md generated successfully.")

Your fully detailed Markdown file for the **Predictive Maintenance** project is ready!

[file-tag: code-generated-file-0-1776341009783937561]

You can download this `.md` file and upload it directly to the root of your GitHub r
