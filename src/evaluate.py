import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from preprocessing import split_and_preprocess

def evaluate_model(model, X_test, y_test, feature_names, output_dir):
    """
    Evaluate the model using various metrics and save visualizations.
    """
    print("\nStarting Model Evaluation...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 2. ROC Curve
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
    # 3. Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Take top 10 features
        top_n = 10
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_features, hue=top_features, palette='viridis', legend=False)
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
    elif hasattr(model, 'coef_'):
        # For Logistic Regression
        importances = np.abs(model.coef_[0])
        indices = np.argsort(importances)[::-1]
        
        # Take top 10 features
        top_n = 10
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_features, hue=top_features, palette='viridis', legend=False)
        plt.title('Top 10 Feature Importances (Absolute Coefficients)')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()

    print(f"\nEvaluation completed. Visualizations saved in {output_dir}")

if __name__ == "__main__":
    dataset_path = os.path.join('..', 'dataset', 'Telco-Customer-Churn.csv')
    models_path = os.path.join('..', 'models')
    outputs_path = os.path.join('..', 'outputs')
    
    # Get processed test data and feature names
    _, X_test, _, y_test, feature_names = split_and_preprocess(dataset_path, models_path)
    
    # Load the best model
    best_model_path = os.path.join(models_path, 'best_model.pkl')
    if not os.path.exists(best_model_path):
        print("Best model not found. Please run train_model.py first.")
    else:
        best_model = joblib.load(best_model_path)
        evaluate_model(best_model, X_test, y_test, feature_names, outputs_path)