import os
import sys

# Add src to python path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from eda import run_eda
from preprocessing import split_and_preprocess
from train_model import train_and_compare_models
from evaluate import evaluate_model
from predict import predict_new_customer

def main():
    print("="*50)
    print(" Telecom Customer Churn Prediction Pipeline ")
    print("="*50)
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'dataset', 'Telco-Customer-Churn.csv')
    outputs_path = os.path.join(base_dir, 'outputs')
    models_path = os.path.join(base_dir, 'models')
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please place the 'Telco-Customer-Churn.csv' file in the dataset directory.")
        return

    # 1. Exploratory Data Analysis
    print("\n--- Step 1: Exploratory Data Analysis ---")
    run_eda(dataset_path, outputs_path)
    
    # 2. Data Preprocessing
    print("\n--- Step 2: Data Preprocessing ---")
    X_train, X_test, y_train, y_test, feature_names = split_and_preprocess(dataset_path, models_path)
    
    # 3. Model Training
    print("\n--- Step 3: Model Training ---")
    best_model, best_model_name = train_and_compare_models(X_train, y_train, X_test, y_test, models_path)
    
    # 4. Model Evaluation
    print("\n--- Step 4: Model Evaluation ---")
    evaluate_model(best_model, X_test, y_test, feature_names, outputs_path)
    
    # 5. Prediction
    print("\n--- Step 5: New Customer Prediction ---")
    sample_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'No',
        'MultipleLines': 'No phone service',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 29.85,
        'TotalCharges': 29.85
    }
    print("Sample Customer Data:", sample_customer)
    predict_new_customer(sample_customer, models_path)
    
    print("\n" + "="*50)
    print(" Pipeline Execution Completed Successfully! ")
    print("="*50)

if __name__ == "__main__":
    main()