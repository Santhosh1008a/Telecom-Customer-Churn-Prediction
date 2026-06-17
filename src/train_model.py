import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import split_and_preprocess

def train_and_compare_models(X_train, y_train, X_test, y_test, model_dir):
    """
    Train multiple models and save the best one based on F1-score.
    """
    print("Starting Model Training...")
    
    # Define models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    best_model_name = ""
    best_model = None
    best_f1_score = 0.0
    
    print("\nModel Comparison:")
    print("-" * 30)
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"{name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score: {f1:.4f}\n")
        
        # Check if this is the best model
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name
            best_model = model
            
    print("-" * 30)
    print(f"Best Model: {best_model_name} with F1-Score: {best_f1_score:.4f}")
    
    # Save the best model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_model_path = os.path.join(model_dir, 'best_model.pkl')
    joblib.dump(best_model, best_model_path)
    print(f"Best model saved at {best_model_path}")
    
    return best_model, best_model_name

if __name__ == "__main__":
    dataset_path = os.path.join('..', 'dataset', 'Telco-Customer-Churn.csv')
    models_path = os.path.join('..', 'models')
    
    # Get processed data
    X_train, X_test, y_train, y_test, _ = split_and_preprocess(dataset_path, models_path)
    
    # Train models and save best
    train_and_compare_models(X_train, y_train, X_test, y_test, models_path)