"""
Time-related utility functions
"""
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pandas as pd


def parse_timestamp(timestamp: str, format: Optional[str] = None) -> datetime:
    """
    Parse timestamp string to datetime object
    
    Args:
        timestamp: Timestamp string (ISO format or custom)
        format: Optional format string for parsing
        
    Returns:
        Datetime object
    """
    if format:
        return datetime.strptime(timestamp, format)
    else:
        # Try ISO format first
        try:
            return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            # Try common formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse timestamp: {timestamp}")


def calculate_time_period(start_time: datetime, end_time: datetime) -> dict:
    """
    Calculate time period statistics
    
    Args:
        start_time: Start datetime
        end_time: End datetime
        
    Returns:
        Dictionary with period statistics
    """
    delta = end_time - start_time
    
    return {
        'total_seconds': delta.total_seconds(),
        'total_minutes': delta.total_seconds() / 60,
        'total_hours': delta.total_seconds() / 3600,
        'total_days': delta.days,
        'total_weeks': delta.days / 7,
        'total_months': delta.days / 30,
        'total_years': delta.days / 365
    }


def get_time_bins(start_time: datetime, 
                 end_time: datetime, 
                 freq: str = 'D') -> pd.DatetimeIndex:
    """
    Get time bins for aggregation
    
    Args:
        start_time: Start datetime
        end_time: End datetime
        freq: Frequency string ('D', 'H', 'W', 'M')
        
    Returns:
        DatetimeIndex with time bins
    """
    return pd.date_range(start=start_time, end=end_time, freq=freq)


def align_timestamps(timestamps: list, freq: str = 'D') -> pd.Series:
    """
    Align timestamps to regular intervals
    
    Args:
        timestamps: List of datetime objects
        freq: Frequency string
        
    Returns:
        Series with aligned timestamps
    """
    df = pd.DataFrame({'timestamp': timestamps})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # Resample to regular intervals
    aligned = df.resample(freq).first()
    
    return aligned.index


def calculate_time_features(timestamp: datetime) -> dict:
    """
    Calculate time-based features from timestamp
    
    Args:
        timestamp: Datetime object
        
    Returns:
        Dictionary with time features
    """
    return {
        'year': timestamp.year,
        'month': timestamp.month,
        'day': timestamp.day,
        'hour': timestamp.hour,
        'minute': timestamp.minute,
        'day_of_week': timestamp.weekday(),  # 0 = Monday
        'day_name': timestamp.strftime('%A'),
        'is_weekend': timestamp.weekday() >= 5,
        'quarter': (timestamp.month - 1) // 3 + 1,
        'week_of_year': timestamp.isocalendar()[1]
    }


def filter_by_time_range(df: pd.DataFrame,
                         time_column: str,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Filter DataFrame by time range
    
    Args:
        df: Input DataFrame
        time_column: Name of time column
        start_time: Start datetime (inclusive)
        end_time: End datetime (inclusive)
        
    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()
    df_filtered[time_column] = pd.to_datetime(df_filtered[time_column])
    
    if start_time:
        df_filtered = df_filtered[df_filtered[time_column] >= start_time]
    
    if end_time:
        df_filtered = df_filtered[df_filtered[time_column] <= end_time]
    
    return df_filtered

