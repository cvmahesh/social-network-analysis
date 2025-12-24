"""
Data manipulation utilities
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any


def clean_dataframe(df: pd.DataFrame, 
                   drop_duplicates: bool = True,
                   drop_na: bool = False,
                   columns_to_drop_na: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Clean DataFrame
    
    Args:
        df: Input DataFrame
        drop_duplicates: Whether to drop duplicate rows
        drop_na: Whether to drop rows with any NA values
        columns_to_drop_na: Specific columns to check for NA
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    if drop_na:
        if columns_to_drop_na:
            df_clean = df_clean.dropna(subset=columns_to_drop_na)
        else:
            df_clean = df_clean.dropna()
    
    return df_clean


def validate_data(df: pd.DataFrame, 
                 required_columns: Optional[List[str]] = None,
                 check_types: Optional[Dict[str, type]] = None) -> bool:
    """
    Validate DataFrame structure and data
    
    Args:
        df: Input DataFrame
        required_columns: List of required column names
        check_types: Dictionary mapping column names to expected types
        
    Returns:
        True if data is valid
        
    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    if check_types:
        for column, expected_type in check_types.items():
            if column in df.columns:
                if not df[column].dtype == expected_type:
                    raise ValueError(
                        f"Column {column} has type {df[column].dtype}, "
                        f"expected {expected_type}"
                    )
    
    return True


def normalize_column(df: pd.DataFrame, column: str, method: str = 'min_max') -> pd.DataFrame:
    """
    Normalize a column in DataFrame
    
    Args:
        df: Input DataFrame
        column: Column name to normalize
        method: Normalization method ('min_max', 'z_score', 'max')
        
    Returns:
        DataFrame with normalized column
    """
    df_norm = df.copy()
    
    if method == 'min_max':
        min_val = df[column].min()
        max_val = df[column].max()
        if max_val > min_val:
            df_norm[column] = (df[column] - min_val) / (max_val - min_val)
        else:
            df_norm[column] = 0.0
    
    elif method == 'z_score':
        mean_val = df[column].mean()
        std_val = df[column].std()
        if std_val > 0:
            df_norm[column] = (df[column] - mean_val) / std_val
        else:
            df_norm[column] = 0.0
    
    elif method == 'max':
        max_val = df[column].max()
        if max_val > 0:
            df_norm[column] = df[column] / max_val
        else:
            df_norm[column] = 0.0
    
    return df_norm


def aggregate_by_time(df: pd.DataFrame, 
                     time_column: str,
                     value_columns: List[str],
                     freq: str = 'D') -> pd.DataFrame:
    """
    Aggregate data by time period
    
    Args:
        df: Input DataFrame
        time_column: Name of time column
        value_columns: Columns to aggregate
        freq: Frequency string ('D' for daily, 'H' for hourly, etc.)
        
    Returns:
        Aggregated DataFrame
    """
    df_agg = df.copy()
    df_agg[time_column] = pd.to_datetime(df_agg[time_column])
    df_agg = df_agg.set_index(time_column)
    
    agg_dict = {col: ['sum', 'mean', 'count'] for col in value_columns}
    result = df_agg.resample(freq).agg(agg_dict)
    
    return result


def export_to_formats(df: pd.DataFrame, base_filename: str, formats: List[str] = ['csv', 'json']):
    """
    Export DataFrame to multiple formats
    
    Args:
        df: Input DataFrame
        base_filename: Base filename (without extension)
        formats: List of formats ('csv', 'json', 'excel', 'parquet')
    """
    for fmt in formats:
        if fmt == 'csv':
            df.to_csv(f"{base_filename}.csv", index=False)
        elif fmt == 'json':
            df.to_json(f"{base_filename}.json", orient='records', indent=2)
        elif fmt == 'excel':
            df.to_excel(f"{base_filename}.xlsx", index=False)
        elif fmt == 'parquet':
            df.to_parquet(f"{base_filename}.parquet", index=False)

