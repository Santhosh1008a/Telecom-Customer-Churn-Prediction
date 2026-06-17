import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def load_data(filepath):
    """
    Load dataset from CSV file.
    """
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocess the dataset:
    - Drop customerID
    - Handle TotalCharges (convert to numeric and fill median)
    - Encode Churn to 1/0
    """
    df = df.copy()
    # Drop unnecessary column
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    # Handle missing values in TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Map Target Variable 'Churn'
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df

def get_preprocessor(df_features):
    """
    Create a ColumnTransformer for numerical and categorical scaling and encoding.
    """
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_features = [col for col in df_features.columns if col not in num_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
        ]
    )
    return preprocessor

def split_and_preprocess(filepath, model_dir):
    """
    Load data, split into train and test sets, apply preprocessing, and save the preprocessor.
    """
    print("Starting Data Preprocessing...")
    df = load_data(filepath)
    df = preprocess_data(df)
    
    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build and fit preprocessor
    preprocessor = get_preprocessor(X)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    
    # Get feature names after encoding for later feature importance extraction
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_features = [col for col in X.columns if col not in num_features]
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
    feature_names = num_features + list(cat_feature_names)
    
    print(f"Data preprocessed successfully. Preprocessor saved at {preprocessor_path}")
    return X_train_processed, X_test_processed, y_train, y_test, feature_names

if __name__ == "__main__":
    dataset_path = os.path.join('..', 'dataset', 'Telco-Customer-Churn.csv')
    models_path = os.path.join('..', 'models')
    split_and_preprocess(dataset_path, models_path)