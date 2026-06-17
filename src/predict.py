import os
import joblib
import pandas as pd

def load_artifacts(models_dir):
    """
    Load the saved preprocessor and the best model.
    """
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    model_path = os.path.join(models_dir, 'best_model.pkl')
    
    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model or Preprocessor not found. Please train the model first.")
        
    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    
    return preprocessor, model

def predict_new_customer(customer_data, models_dir):
    """
    Predict churn for a new customer.
    
    Parameters:
    - customer_data (dict): Dictionary containing customer features.
    - models_dir (str): Directory containing the saved model and preprocessor.
    """
    preprocessor, model = load_artifacts(models_dir)
    
    # Convert dictionary to DataFrame
    df = pd.DataFrame([customer_data])
    
    # Preprocess
    df_processed = preprocessor.transform(df)
    
    # Predict
    prediction = model.predict(df_processed)
    
    # Output result
    if prediction[0] == 1:
        print("\nResult: Customer will churn")
    else:
        print("\nResult: Customer will not churn")

if __name__ == "__main__":
    models_path = os.path.join('..', 'models')
    
    # Sample new customer data
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
    
    print("Predicting for a sample customer...")
    predict_new_customer(sample_customer, models_path)