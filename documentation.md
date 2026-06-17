# Telecom Customer Churn Prediction: Detailed Documentation

## 1. Project Overview
Customer churn is a critical metric for telecommunications companies. This project implements a fully automated end-to-end Machine Learning pipeline to predict whether a customer will churn based on their demographic, service, and account attributes. It serves as a comprehensive Data Science portfolio piece suitable for internship evaluation.

## 2. Methodology

### 2.1 Exploratory Data Analysis (EDA)
- **Visualizing Churn:** Understanding the baseline churn rate and dataset imbalance.
- **Categorical Analyses:** Visualizing churn rates grouped by Gender, Contract Type, and Internet Service. It was observed that Month-to-month contracts have substantially higher churn rates.
- **Numerical Analyses:** Visualizing Tenure and Monthly Charges. High monthly charges and low tenure are heavily correlated with churn.
- **Correlation Mapping:** Generating a correlation heatmap using `sns.heatmap` to verify multilinearity between numerical variables.

### 2.2 Preprocessing
Data preprocessing handles missing or erratic values to ensure models train properly.
- **Data Imputation:** Blank entries in `TotalCharges` were coerced to NaNs and filled using the median total charge.
- **Feature Engineering:** `customerID` was dropped as it carries no predictive power.
- **Encoding & Scaling:** We constructed an `sklearn.compose.ColumnTransformer`. Categorical features underwent `OneHotEncoder(drop='first')` to prevent the dummy variable trap. Numerical features like `TotalCharges`, `MonthlyCharges`, and `tenure` underwent `StandardScaler()` for consistent scale, vital for Logistic Regression distance calculations.

### 2.3 Model Training
Data was split 80/20 (Train/Test) using `train_test_split(stratify=y)` to maintain the churn ratio distribution in both sets. Three baseline models were trained:
1. **Logistic Regression:** A generalized linear model ideal for binary classification tasks.
2. **Decision Tree Classifier:** A non-linear tree-based structure.
3. **Random Forest Classifier:** An ensemble bagging method for variance reduction.

### 2.4 Model Evaluation
The primary metric used for comparison was the **F1-Score**. Accuracy alone is misleading on imbalanced datasets (such as a 73% non-churn vs 27% churn split). The final results achieved were:
- **Logistic Regression:** Accuracy: 0.8055 | F1-Score: 0.6029 *(Selected as Best)*
- **Random Forest:** Accuracy: 0.7857 | F1-Score: 0.5493
- **Decision Tree:** Accuracy: 0.7410 | F1-Score: 0.5047

Additional evaluated items for the best model:
- **Confusion Matrix:** Shows the True Positives, True Negatives, False Positives, and False Negatives.
- **ROC Curve:** Plotting the True Positive Rate against the False Positive Rate at various threshold settings.
- **Feature Importance:** Extracting coefficients from Logistic Regression.

## 3. Project Architecture

```
Telecom-Customer-Churn-Prediction/
│
├── dataset/                     <- Data folder (e.g., Telco-Customer-Churn.csv)
├── notebooks/                   <- Notebooks folder (for experimenting if needed)
├── src/                         <- Source code containing the core logic
│   ├── eda.py                   <- Exploratory Data Analysis generation script
│   ├── preprocessing.py         <- Data cleaning and transformation logic
│   ├── train_model.py           <- Model training and comparison logic
│   ├── evaluate.py              <- Model metric evaluation and visualization logic
│   └── predict.py               <- Inference logic for new customer samples
├── models/                      <- Saved models and preprocessors (e.g., .pkl)
├── outputs/                     <- Saved visualizations (e.g., heatmaps, ROC)
├── screenshots/                 <- Output screenshots folder
├── requirements.txt             <- Python dependencies specification
├── main.py                      <- Main execution orchestration file
├── README.md                    <- Concise project summary and instructions
└── documentation.md             <- Comprehensive documentation (this file)
```

## 4. Results & Screenshots
All visualization charts are exported in `.png` format to the `/outputs` directory.
- `churn_distribution.png`: Demonstrates the target variable class imbalance.
- `contract_type_vs_churn.png`: Proves shorter-term contracts lead to higher churn.
- `roc_curve.png`: Shows an AUC metric highlighting the classifier's ability to separate classes.
- `feature_importance.png`: Visual representation of which encoded variables most affect the logistic regression classification.

*(Export this file to PDF as required to embed or attach these screenshots manually as per standard formatting needs)*

## 5. Future Improvements
- Improvise hyperparameter tuning utilizing `GridSearchCV` on tree ensembles to prevent underfitting.
- Apply `SMOTE` (Synthetic Minority Over-sampling Technique) inside a custom `imblearn` pipeline to synthetically increase the churn class size during training, significantly improving the Recall and F1-Scores.