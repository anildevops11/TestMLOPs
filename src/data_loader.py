"""
Data loading and validation module for iris dataset.

This module provides functions to load iris data from CSV files with fallback
to sklearn's built-in dataset, and validate the data structure and content.
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = Path("data/iris.csv")
REQUIRED_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
VALID_SPECIES_NAMES = ["setosa", "versicolor", "virginica"]
VALID_SPECIES_INDICES = [0, 1, 2]
MIN_ROWS = 10


def load_iris_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load iris dataset with fallback mechanism.
    
    Attempts to load data from CSV file first. If the file is empty, missing,
    or invalid, falls back to sklearn's built-in iris dataset.
    
    Returns:
        tuple: (features_df, target_series) where features_df contains the
               4 feature columns and target_series contains the species labels
    
    Raises:
        ValueError: If data validation fails for both CSV and sklearn data
    """
    # Try loading from CSV first
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        try:
            logger.info(f"Attempting to load data from {DATA_PATH}")
            df = pd.read_csv(DATA_PATH)
            
            # Validate the dataframe
            validate_iris_dataframe(df)
            
            # Extract features and target
            features = df[REQUIRED_COLUMNS]
            
            # Handle both 'species' and 'target' column names
            if 'species' in df.columns:
                target = df['species']
            elif 'target' in df.columns:
                target = df['target']
            else:
                raise ValueError("CSV must contain either 'species' or 'target' column")
            
            logger.info(f"Successfully loaded {len(df)} rows from CSV")
            return features, target
            
        except Exception as e:
            logger.warning(f"Failed to load CSV: {e}. Falling back to sklearn dataset")
    else:
        logger.warning(f"CSV file is empty or missing at {DATA_PATH}. Using sklearn dataset")
    
    # Fallback to sklearn dataset
    logger.info("Loading iris dataset from sklearn")
    iris = load_iris()
    
    # Create DataFrame with proper column names
    features = pd.DataFrame(
        iris.data,
        columns=REQUIRED_COLUMNS
    )
    target = pd.Series(iris.target, name='target')
    
    logger.info(f"Successfully loaded {len(features)} rows from sklearn")
    return features, target


def validate_iris_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate iris dataset structure and content.
    
    Checks that the dataframe has the correct structure:
    - Exactly 5 columns with correct names
    - All feature columns contain numeric values
    - Species/target column contains only valid iris species
    - No null values
    - At least minimum number of rows
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If validation fails with descriptive message
    """
    # Check if dataframe is empty
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check minimum number of rows
    if len(df) < MIN_ROWS:
        raise ValueError(f"DataFrame must have at least {MIN_ROWS} rows, got {len(df)}")
    
    # Check for null values
    if df.isnull().any().any():
        null_columns = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"DataFrame contains null values in columns: {null_columns}")
    
    # Check number of columns (must be exactly 5)
    if len(df.columns) != 5:
        raise ValueError(
            f"DataFrame must have exactly 5 columns, got {len(df.columns)}: {list(df.columns)}"
        )
    
    # Check for required feature columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Required columns are: {REQUIRED_COLUMNS + ['species or target']}"
        )
    
    # Check that either 'species' or 'target' column exists
    has_species = 'species' in df.columns
    has_target = 'target' in df.columns
    
    if not has_species and not has_target:
        raise ValueError(
            "DataFrame must contain either 'species' or 'target' column. "
            f"Found columns: {list(df.columns)}"
        )
    
    # Validate feature columns are numeric
    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Column '{col}' must contain numeric values, "
                f"got dtype: {df[col].dtype}"
            )
    
    # Validate species/target column contains only valid values
    target_col = 'species' if has_species else 'target'
    unique_values = df[target_col].unique()
    
    # Check if values are species names (strings)
    if pd.api.types.is_string_dtype(df[target_col]):
        invalid_species = [
            val for val in unique_values 
            if val not in VALID_SPECIES_NAMES
        ]
        if invalid_species:
            raise ValueError(
                f"Invalid species names found: {invalid_species}. "
                f"Valid species are: {VALID_SPECIES_NAMES}"
            )
    # Check if values are numeric indices
    elif pd.api.types.is_numeric_dtype(df[target_col]):
        invalid_indices = [
            val for val in unique_values 
            if val not in VALID_SPECIES_INDICES
        ]
        if invalid_indices:
            raise ValueError(
                f"Invalid species indices found: {invalid_indices}. "
                f"Valid indices are: {VALID_SPECIES_INDICES} (0=setosa, 1=versicolor, 2=virginica)"
            )
    else:
        raise ValueError(
            f"Column '{target_col}' must be either string (species names) "
            f"or numeric (species indices), got dtype: {df[target_col].dtype}"
        )
    
    logger.info("DataFrame validation passed")
    return True
