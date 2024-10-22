# Readmission Risk Prediction

## Problem Statement

Hospital readmission is a significant concern for healthcare systems, leading to increased costs and patient complications. Accurately predicting which patients are at higher risk of readmission can enable healthcare providers to proactively offer personalized care and reduce the likelihood of readmission. This project aims to analyze comprehensive patient data, including demographics, medical history, vital signs, and treatment outcomes, to predict the risk of hospital readmission. The goal is to equip healthcare providers with predictive insights, allowing them to identify high-risk individuals and develop personalized care plans that prevent potential readmissions.

## Model Training and Data Preparation

### Algorithms:
- **XGBoost**: Used for its ability to handle high-dimensional data efficiently and its strong performance on classification tasks.
- **Logistic Regression**: Applied with both **L1** (Lasso) and **L2** (Ridge) regularization to prevent overfitting and improve model generalization.

### Data Handling:
- **Data Split**: 80% of the data was used for training, and 20% for testing.
- **Preprocessing Steps**:
  - Handling missing values using appropriate imputation methods.
  - Normalizing data to ensure uniform scaling across features.
  - Feature engineering to enhance model performance by creating meaningful features.

### Ensemble Approach:
- An ensemble approach was employed by combining the strengths of XGBoost and Logistic Regression to improve the overall model performance.

## Evaluation and Metrics:
- **Metrics Used**:
  - Accuracy
  - Sensitivity (Recall)
  - Specificity
  - Precision
  - F1 Score
  - AUC-ROC
  - Confusion Matrix

- **Performance**:
  - Achieved **91% accuracy** in predicting hospital readmission risk.

## Evaluation and Results

- **Output**:
  - The model identifies high-risk and low-risk patients with their respective probability of readmission.
  
- **Root Cause Analysis**:
  - The system provides insights into potential causes for readmission for each patient based on the features contributing to the prediction.

## Deployment:
- The model was integrated into a frontend application, allowing for real-time predictions.
- The trained model was saved using **Pickle** for efficient reuse and future predictions.

## Risk Identification:
- The system highlights patients at risk and suggests tailored interventions to prevent readmissions.

## Dataset Link: https://datadryad.org/stash/dataset/doi:10.5061/dryad.70rxwdbxw
## procedure:
- create account on twillo and firebase and download the key as json and upload it in the folder.
- Run - dash_app.py to access the visulation on the patients reports
- run app_fin.py
for more queries message me on gmail: **deeparamya2004@gmail.com**

