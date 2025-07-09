
import pandas as pd


def load_and_clean_data(file_path) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: There was a parsing error.")
        return None
    
def print_basic_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    """
    
    if df is not None:
        print("Basic Information:")
        print(df.info())
        
        print("\nDescriptive Statistics:")
        print(df.describe()) 
    else:
        print("No data to display.")

def clean_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by:
    - Converting TotalCharges to numeric
    - Dropping customerID column
    - Handling missing values (drop rows with NaNs)
    - Removing duplicate rows

    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    if df is None or df.empty:
        print("[WARN] Provided DataFrame is None or empty. Nothing to preprocess.")
        return df

    # Convert TotalCharges to numeric, coerce errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")
    
    # Drop customerID column if it exists
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Report before dropping
    missing_before = df.isnull().sum().sum()
    duplicates_before = df.duplicated().sum()
    print(f"[INFO] Missing values before: {missing_before}")
    print(f"[INFO] Duplicate rows before: {duplicates_before}")

    # Drop missing values
    df.dropna(inplace=True)

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Report after dropping
    missing_after = df.isnull().sum().sum()
    duplicates_after = df.duplicated().sum()
    print(f"[INFO] Missing values after: {missing_after}")
    print(f"[INFO] Duplicate rows after: {duplicates_after}")

    print(f"[INFO] Final dataset shape: {df.shape}")

    return df


