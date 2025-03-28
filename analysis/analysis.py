# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import csv

# Set plot styles
sns.set_theme(style="whitegrid")  # Using sns.set_theme() instead of plt.style.use('seaborn')
%matplotlib inline  
# This is valid in Jupyter notebooks

# For displaying all columns in pandas dataframes
pd.set_option('display.max_columns', None)

# %% Data Loading Functions
# Define functions to detect delimiters and load data with proper encoding

def detect_delimiter(file_path, encoding='latin-1', bytes_to_read=4096):
    """
    Detect the delimiter used in a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    encoding : str, default='latin-1'
        Encoding to use for reading the file
    bytes_to_read : int, default=4096
        Number of bytes to read for detection
        
    Returns:
    --------
    str
        Detected delimiter character
    """
    try:
        with open(file_path, 'r', encoding=encoding) as csvfile:
            header = csvfile.readline()
            # Common delimiters to check
            for delimiter in [',', ';', '\t', '|']:
                if delimiter in header:
                    return delimiter
        return ','  # Default to comma if nothing found
    except Exception as e:
        print(f"Error detecting delimiter in {file_path}: {e}")
        return ','  # Default to comma on error

def load_csv_data(file_path, encoding='latin-1', **kwargs):
    """
    Load a CSV file into a pandas DataFrame with encoding handling and delimiter detection.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    encoding : str, default='latin-1'
        Encoding to use for reading the CSV file
    **kwargs : 
        Additional arguments to pass to pd.read_csv()
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the CSV data
    """
    try:
        # First try with delimiter detection and specified encoding
        detected_delimiter = detect_delimiter(file_path, encoding=encoding)
        print(f"Detected delimiter: '{detected_delimiter}' for {file_path}")
        
        # Try with detected delimiter
        df = pd.read_csv(file_path, encoding=encoding, delimiter=detected_delimiter, 
                         quoting=csv.QUOTE_MINIMAL, **kwargs)
        print(f"Successfully loaded {file_path} with {encoding} encoding and '{detected_delimiter}' delimiter")
        return df
    except UnicodeDecodeError:
        # If encoding fails, try with cp1252 (common in Brazil/Portugal)
        try:
            detected_delimiter = detect_delimiter(file_path, encoding='cp1252')
            df = pd.read_csv(file_path, encoding='cp1252', delimiter=detected_delimiter, 
                             quoting=csv.QUOTE_MINIMAL, **kwargs)
            print(f"Successfully loaded {file_path} with cp1252 encoding and '{detected_delimiter}' delimiter")
            return df
        except UnicodeDecodeError:
            # Last resort: use utf-8 with error replacement
            print(f"Warning: Encoding issues with {file_path}, using utf-8 with replacement characters")
            detected_delimiter = detect_delimiter(file_path, encoding='utf-8')
            df = pd.read_csv(file_path, encoding='utf-8', errors='replace', 
                             delimiter=detected_delimiter, quoting=csv.QUOTE_MINIMAL, **kwargs)
            return df
    except Exception as e:
        print(f"Error loading {file_path} with detected delimiter: {e}")
        # Try with more flexible parsing options
        try:
            # For newer pandas versions
            df = pd.read_csv(file_path, encoding=encoding, delimiter=';', 
                             on_bad_lines='skip', **kwargs)
            print(f"Successfully loaded {file_path} with errors skipped (using ';' delimiter)")
            return df
        except Exception as e1:
            try:
                # For older pandas versions
                df = pd.read_csv(file_path, encoding=encoding, delimiter=';', 
                                 error_bad_lines=False, warn_bad_lines=True, **kwargs)
                print(f"Successfully loaded {file_path} with errors skipped (using ';' delimiter)")
                return df
            except Exception as e2:
                print(f"All attempts to load {file_path} failed: {e2}")
                return None

def load_data(file_path, **kwargs):
    """Load data from different file formats based on extension"""
    if file_path.endswith('.csv'):
        return load_csv_data(file_path, **kwargs)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path, **kwargs)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path, **kwargs)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path, **kwargs)
    elif file_path.endswith('.pickle') or file_path.endswith('.pkl'):
        return pd.read_pickle(file_path, **kwargs)
    elif file_path.endswith('.hdf') or file_path.endswith('.h5'):
        return pd.read_hdf(file_path, **kwargs)
    else:
        print(f"Unsupported file format: {file_path}")
        return None

def list_available_datasets(directory="/home/hpfeffer/development/poc-rhoai/synth-data-sdv/datasets", recursive=True):
    """
    List all CSV files available in the specified directory and its subfolders.
    
    Parameters:
    -----------
    directory : str, default="/home/hpfeffer/development/poc-rhoai/synth-data-sdv/datasets"
        Directory to search for CSV files
    recursive : bool, default=True
        Whether to search recursively in subfolders
        
    Returns:
    --------
    list
        List of CSV file paths
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []
    
    csv_files = []
    
    if recursive:
        # Walk through all subdirectories
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
    else:
        # Only look in the top directory
        csv_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                    if f.endswith('.csv') and os.path.isfile(os.path.join(directory, f))]
    
    if not csv_files:
        print(f"No CSV files found in {directory}" + (" and its subfolders" if recursive else ""))
    else:
        print(f"Found {len(csv_files)} CSV files in {directory}" + (" and its subfolders" if recursive else "") + ":")
        for file in csv_files:
            print(f"  - {file}")
    
    return csv_files

# %% Load Datasets
# Use the proper path based on the existing codebase

# Path to datasets directory
data_dir = "/home/hpfeffer/development/poc-rhoai/synth-data-sdv/datasets"  # Use the path from the codebase
print(f"Searching for datasets in: {data_dir}")

# Get list of data files
csv_files = list_available_datasets(data_dir)

# Load each dataset into a dictionary of dataframes
datasets = {}
for file_path in csv_files:
    file_name = os.path.basename(file_path)
    name = os.path.splitext(file_name)[0]
    datasets[name] = load_data(file_path)
    print(f"Loaded {file_name}, shape: {datasets[name].shape if datasets[name] is not None else 'N/A'}")

# %% Organize datasets into a hierarchical structure
# Create nested dictionary based on directory structure
organized_datasets = {}

for file_path in csv_files:
    # Get relative path from the base directory
    rel_path = os.path.relpath(file_path, data_dir)
    # Split path components
    path_components = rel_path.split(os.sep)
    
    # Create nested dictionary structure based on folder hierarchy
    current_level = organized_datasets
    for component in path_components[:-1]:  # Navigate through folders
        if component not in current_level:
            current_level[component] = {}
        current_level = current_level[component]
    
    # Add the DataFrame at the appropriate level
    file_name = os.path.basename(file_path).replace('.csv', '')
    current_level[file_name] = load_data(file_path)

# %% Basic Data Overview
# Display basic information about each dataset
for name, df in datasets.items():
    if df is not None:
        print(f"\n=== {name} Dataset ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nSample data:")
        display(df.head())  # Using Jupyter's display function
        
        # Memory usage information
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # Convert to MB
        print(f"Memory usage: {memory_usage:.2f} MB")

# %% Missing Values Analysis
# Analyze missing values in each dataset
for name, df in datasets.items():
    if df is not None:
        print(f"\n=== Missing Values in {name} Dataset ===")
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        
        missing_info = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_percent
        })
        
        # Only show columns with missing values
        missing_info = missing_info[missing_info['Missing Values'] > 0]
        
        if len(missing_info) > 0:
            display(missing_info.sort_values('Missing Values', ascending=False))
            
            # Visualize missing values if there are any
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
            plt.title(f'Missing Values in {name} Dataset')
            plt.tight_layout()
            plt.show()
        else:
            print("No missing values found.")

# %% Data Preprocessing
# Function for basic preprocessing
def preprocess_data(df):
    """Basic preprocessing steps for numerical data"""
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Handle missing values (simple imputation)
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # For categorical columns, fill with mode
    cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else "Unknown")
    
    return df_processed

# Process each dataset
processed_datasets = {}
for name, df in datasets.items():
    if df is not None:
        processed_datasets[name] = preprocess_data(df)
        print(f"Preprocessed {name} dataset")

# %% Display the organized datasets structure
organized_datasets.keys()
