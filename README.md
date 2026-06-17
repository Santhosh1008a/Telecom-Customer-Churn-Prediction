# Telecom Customer Churn Prediction

## Objective
The objective of this project is to build an end-to-end Machine Learning pipeline to predict customer churn in the telecommunications industry. Predicting customer churn helps companies take proactive steps to retain customers, thereby increasing revenue.

## Dataset Information
This project uses the **IBM Telco Customer Churn dataset**. It contains information about a fictitious telco company that provided home phone and internet services to 7043 customers in California in Q3. It includes details about customer demographics, account information, services they signed up for, and whether they left the company (churned) within the last month.

## Technologies Used
- **Python 3.11**
- **Pandas & NumPy:** Data manipulation and analysis.
- **Matplotlib & Seaborn:** Data visualization.
- **Scikit-Learn:** Machine learning, preprocessing, metrics, and evaluation.
- **Joblib:** Saving and loading the best models and preprocessors.

## Workflow
1. **Exploratory Data Analysis (EDA):** Visualizing distributions, handling outliers, and finding correlations to understand the factors driving churn.
2. **Data Preprocessing:** Cleaning missing data (like blank values in TotalCharges), encoding categorical variables (One-Hot Encoding), and scaling numerical variables (Standard Scaler).
3. **Model Training:** Training and comparing Logistic Regression, Decision Tree, and Random Forest Classifier models. 
4. **Model Evaluation:** Computing metrics such as Accuracy, Precision, Recall, F1-Score, building confusion matrices, generating classification reports, plotting ROC-AUC curves, and determining feature importance.
5. **Prediction:** Designing an inference script capable of taking new customer inputs and classifying whether they will churn or not.

## Results
- Evaluated models:
  - **Logistic Regression:** Accuracy: ~0.8055, F1-Score: ~0.6029
  - **Random Forest:** Accuracy: ~0.7857, F1-Score: ~0.5493
  - **Decision Tree:** Accuracy: ~0.7410, F1-Score: ~0.5047
- The **Logistic Regression** model proved to be the most robust based on the F1-Score metric and was selected as the best model.
- Key factors influencing churn included specific contract types (e.g., month-to-month), tenure, and total charges.

## Future Improvements
- **Hyperparameter Tuning:** Applying GridSearchCV or RandomizedSearchCV to optimize models further, especially the Random Forest.
- **Class Imbalance Handling:** Using techniques like SMOTE or adjusting class weights to improve recall for the churned minority class.
- **Advanced Ensembles:** Trying advanced algorithms like XGBoost, LightGBM, or CatBoost.
- **Web App / API Deployment:** Creating a REST API (using FastAPI or Flask) or an interactive web app (using Streamlit) for user-friendly inference.
- **Pipeline Deployment:** Containerizing the process using Docker.

## Setup Instructions
1. Place the dataset `Telco-Customer-Churn.csv` in the `dataset/` directory.
2. Run `pip install -r requirements.txt`.
3. Run `python main.py` from the root directory to execute the end-to-end pipeline.

## Outputs
All visualizations, including histograms, heatmaps, and ROC curves, are saved inside the `outputs/` folder. Models are dumped inside the `models/` folder.