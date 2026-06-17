# Telecom Customer Churn Prediction

## Internship Details

* **Intern ID:** CTTS057
* **Project Name:** Telecom Customer Churn Prediction

## Project Scope

The objective of this project is to build an end-to-end Machine Learning pipeline to predict customer churn in the telecommunications industry. Predicting customer churn helps companies take proactive steps to retain customers, thereby increasing revenue.

## Dataset Information

This project uses the **IBM Telco Customer Churn dataset**. It contains information about a fictitious telecom company that provided home phone and internet services to 7043 customers. The dataset includes customer demographics, account information, subscribed services, and churn status.

## Technologies Used

* Python 3.11
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-Learn
* Joblib

## Workflow

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Model Training
4. Model Evaluation
5. Prediction

## Results

### Models Evaluated

* Logistic Regression – Accuracy: ~0.8055, F1-Score: ~0.6029
* Random Forest – Accuracy: ~0.7857, F1-Score: ~0.5493
* Decision Tree – Accuracy: ~0.7410, F1-Score: ~0.5047

### Best Model

Logistic Regression achieved the best performance and was selected as the final model.

### Key Insights

* Customers with month-to-month contracts have higher churn rates.
* Longer customer tenure is associated with lower churn.
* Total charges and monthly charges significantly influence customer churn.

## Future Improvements

* Hyperparameter tuning using GridSearchCV
* Handling class imbalance using SMOTE
* Trying XGBoost, LightGBM, and CatBoost
* Deploying with Streamlit or FastAPI
* Dockerizing the pipeline

## Setup Instructions

1. Place `Telco-Customer-Churn.csv` inside the `dataset/` folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the project:

```bash
python main.py
```

## Outputs

* EDA Visualizations
* Correlation Heatmap
* Confusion Matrix
* ROC Curve
* Feature Importance Graph
* Trained Model (`best_model.pkl`)
* Saved Preprocessor (`preprocessor.pkl`)
