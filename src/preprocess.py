import pandas as pd
import numpy as np
from datetime import datetime   
from sklearn.model_selection import train_test_split

def load_data(path='../data/raw/Telco-Customer-Churn.csv'):
    """Load the Telco Customer Churn dataset."""

    df = pd.read_csv(path)
    return df

def clean_data(df):
    """Clean and preprocess the Telco dataset."""
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows with missing TotalCharges
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)

    # Drop irrelevant column
    df.drop(columns=['customerID'], inplace=True)

    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def encode_features(df):
    """One-hot encode categorical variables."""

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df_encoded

def split_data(df_encoded, test_size=0.2, random_state=42):
    """Split features and labels into train/test sets."""
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def preprocess_pipeline(path='../data/raw/Telco-Customer-Churn.csv'):
    """Full preprocessing pipeline returning train/test sets."""

    df = load_data(path)
    df_cleaned = clean_data(df)
    df_encoded = encode_features(df_cleaned)
    X_train, X_test, y_train, y_test = split_data(df_encoded)
   
    return X_train, X_test, y_train, y_test


def save_processed_data(X, y, folder='../data/processed/', prefix='df_encoded_resampled'):
    """
    Save processed dataset (X and y combined) to CSV with current date.
    
    Parameters:
    - X: pd.DataFrame, feature matrix
    - y: pd.Series or np.array, target variable
    - folder: target folder to save file
    - prefix: filename prefix
    
    Returns:
    - filename: full path of saved file
    """
    # Combine X and y
    df_final = pd.DataFrame(X, columns=X.columns)
    df_final['Churn'] = y

    # Create filename with current date
    date_str = datetime.today().strftime('%Y%m%d')
    filename = f"{folder}.csv"

    # Save to CSV
    df_final.to_csv(filename, index=False)
    print(f"✅ Saved processed data to: {filename}")
    
    return filename
