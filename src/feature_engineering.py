import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

### --- ENCODING FUNCTIONS ---

def encode_binary_columns(df, columns):
    """
    Encode binary categorical columns (e.g., Yes/No) as 0/1.
    """
    df_copy = df.copy()
    for col in columns:
        df_copy[col] = df_copy[col].map({'Yes': 1, 'No': 0})
    return df_copy

def encode_label_columns(df, columns):
    """
    Label encode categorical columns with more than two categories.
    """
    df_copy = df.copy()
    le = LabelEncoder()
    for col in columns:
        df_copy[col] = le.fit_transform(df_copy[col])
    return df_copy

def one_hot_encode_columns(df, columns):
    """
    One-hot encode selected categorical columns.
    """
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
    return df_encoded

### --- SCALING FUNCTIONS ---

def scale_numeric_columns(df, columns):
    """
    Standard scale numeric columns (zero mean, unit variance).
    """
    df_copy = df.copy()
    scaler = StandardScaler()
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy

### --- FEATURE CREATION FUNCTIONS ---

def create_tenure_group(df):
    """
    Create a new feature grouping customers by tenure.
    """
    df_copy = df.copy()
    def tenure_bin(tenure):
        if tenure <= 12:
            return '0-12 Months'
        elif tenure <= 24:
            return '12-24 Months'
        elif tenure <= 48:
            return '24-48 Months'
        elif tenure <= 60:
            return '48-60 Months'
        else:
            return '60+ Months'
    df_copy['TenureGroup'] = df_copy['tenure'].apply(tenure_bin)
    return df_copy

def add_total_services(df):
    """
    Add a feature counting total services subscribed by customer.
    """
    df_copy = df.copy()
    services = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df_copy['TotalServices'] = df_copy[services].apply(lambda row: sum(row == 'Yes'), axis=1)
    return df_copy
