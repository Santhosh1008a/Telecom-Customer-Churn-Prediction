import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(filepath):
    """
    Load the Telco Customer Churn dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    return df


def clean_total_charges(df):
    """
    Clean the 'TotalCharges' column by coercing errors and filling NaNs.
    """
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill missing values with median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df


def plot_churn_distribution(df, output_dir):
    """
    Plot and save churn distribution.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn', hue='Churn', palette='Set2', legend=False)
    plt.title('Churn Distribution')
    plt.savefig(os.path.join(output_dir, 'churn_distribution.png'))
    plt.close()


def plot_gender_distribution(df, output_dir):
    """
    Plot and save gender distribution.
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='gender', hue='Churn', palette='Set2')
    plt.title('Gender Distribution by Churn')
    plt.savefig(os.path.join(output_dir, 'gender_distribution.png'))
    plt.close()


def plot_contract_type_vs_churn(df, output_dir):
    """
    Plot and save contract type vs churn.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette='Set2')
    plt.title('Contract Type vs Churn')
    plt.savefig(os.path.join(output_dir, 'contract_type_vs_churn.png'))
    plt.close()


def plot_monthly_charges_distribution(df, output_dir):
    """
    Plot and save monthly charges distribution.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple='stack', palette='Set2', kde=True)
    plt.title('Monthly Charges Distribution by Churn')
    plt.savefig(os.path.join(output_dir, 'monthly_charges_distribution.png'))
    plt.close()


def plot_tenure_distribution(df, output_dir):
    """
    Plot and save tenure distribution.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', palette='Set2', kde=True)
    plt.title('Tenure Distribution by Churn')
    plt.savefig(os.path.join(output_dir, 'tenure_distribution.png'))
    plt.close()


def plot_internet_service_vs_churn(df, output_dir):
    """
    Plot and save Internet Service vs Churn.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='InternetService', hue='Churn', palette='Set2')
    plt.title('Internet Service vs Churn')
    plt.savefig(os.path.join(output_dir, 'internet_service_vs_churn.png'))
    plt.close()


def plot_correlation_heatmap(df, output_dir):
    """
    Plot and save correlation heatmap for numerical features.
    """
    # Replace Churn with 1/0 for correlation calculation
    df_copy = df.copy()
    df_copy['Churn'] = df_copy['Churn'].map({'Yes': 1, 'No': 0})
    
    # Get numerical features only
    num_df = df_copy[['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']]
    corr = num_df.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'heatmap.png'))
    plt.close()


def run_eda(filepath, output_dir):
    """
    Run the entire EDA pipeline.
    """
    print("Starting EDA...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = load_dataset(filepath)
    df = clean_total_charges(df)
    
    plot_churn_distribution(df, output_dir)
    plot_gender_distribution(df, output_dir)
    plot_contract_type_vs_churn(df, output_dir)
    plot_monthly_charges_distribution(df, output_dir)
    plot_tenure_distribution(df, output_dir)
    plot_internet_service_vs_churn(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    
    print(f"EDA completed. Visualizations saved in {output_dir}")

if __name__ == '__main__':
    dataset_path = os.path.join('..', 'dataset', 'Telco-Customer-Churn.csv')
    outputs_path = os.path.join('..', 'outputs')
    run_eda(dataset_path, outputs_path)