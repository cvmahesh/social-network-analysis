"""
Metrics calculation and normalization utilities
"""
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize_metrics(metrics_dict: Dict[str, Union[float, int]], 
                     method: str = 'min_max') -> Dict[str, float]:
    """
    Normalize metrics dictionary to [0, 1] range
    
    Args:
        metrics_dict: Dictionary mapping metric names to values
        method: Normalization method ('min_max', 'z_score', 'max')
        
    Returns:
        Dictionary with normalized values
    """
    if not metrics_dict:
        return {}
    
    values = np.array(list(metrics_dict.values()))
    
    if method == 'min_max':
        min_val = values.min()
        max_val = values.max()
        if max_val > min_val:
            normalized_values = (values - min_val) / (max_val - min_val)
        else:
            normalized_values = np.zeros_like(values)
    
    elif method == 'z_score':
        mean_val = values.mean()
        std_val = values.std()
        if std_val > 0:
            normalized_values = (values - mean_val) / std_val
            # Scale to [0, 1] range
            normalized_values = (normalized_values - normalized_values.min()) / (
                normalized_values.max() - normalized_values.min() + 1e-10
            )
        else:
            normalized_values = np.zeros_like(values)
    
    elif method == 'max':
        max_val = values.max()
        if max_val > 0:
            normalized_values = values / max_val
        else:
            normalized_values = np.zeros_like(values)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return dict(zip(metrics_dict.keys(), normalized_values))


def aggregate_metrics(metrics_list: List[Dict[str, float]], 
                     method: str = 'mean') -> Dict[str, float]:
    """
    Aggregate multiple metrics dictionaries
    
    Args:
        metrics_list: List of metrics dictionaries
        method: Aggregation method ('mean', 'sum', 'max', 'min')
        
    Returns:
        Aggregated metrics dictionary
    """
    if not metrics_list:
        return {}
    
    # Get all unique keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    aggregated = {}
    for key in all_keys:
        values = [m.get(key, 0) for m in metrics_list if key in m]
        
        if method == 'mean':
            aggregated[key] = np.mean(values)
        elif method == 'sum':
            aggregated[key] = np.sum(values)
        elif method == 'max':
            aggregated[key] = np.max(values)
        elif method == 'min':
            aggregated[key] = np.min(values)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    return aggregated


def combine_metrics(metrics_dicts: Dict[str, Dict[str, float]],
                   weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Combine multiple metrics dictionaries with weights
    
    Args:
        metrics_dicts: Dictionary mapping metric group names to metrics dicts
        weights: Dictionary mapping metric group names to weights (must sum to 1)
        
    Returns:
        Combined metrics dictionary
    """
    if not metrics_dicts:
        return {}
    
    # Default equal weights
    if weights is None:
        n = len(metrics_dicts)
        weights = {key: 1.0 / n for key in metrics_dicts.keys()}
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    # Get all unique keys
    all_keys = set()
    for metrics in metrics_dicts.values():
        all_keys.update(metrics.keys())
    
    combined = {}
    for key in all_keys:
        combined[key] = sum(
            weights.get(group, 0) * metrics.get(key, 0)
            for group, metrics in metrics_dicts.items()
        )
    
    return combined


def calculate_percentile_rank(values: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate percentile rank for each value
    
    Args:
        values: Dictionary mapping keys to values
        
    Returns:
        Dictionary with percentile ranks (0-1)
    """
    if not values:
        return {}
    
    sorted_values = sorted(values.values())
    n = len(sorted_values)
    
    percentile_ranks = {}
    for key, value in values.items():
        rank = sum(1 for v in sorted_values if v <= value)
        percentile_ranks[key] = rank / n if n > 0 else 0.0
    
    return percentile_ranks


def calculate_correlation(metrics1: Dict[str, float], 
                         metrics2: Dict[str, float]) -> float:
    """
    Calculate correlation between two metrics dictionaries
    
    Args:
        metrics1: First metrics dictionary
        metrics2: Second metrics dictionary
        
    Returns:
        Correlation coefficient
    """
    # Get common keys
    common_keys = set(metrics1.keys()) & set(metrics2.keys())
    
    if len(common_keys) < 2:
        return 0.0
    
    values1 = [metrics1[key] for key in common_keys]
    values2 = [metrics2[key] for key in common_keys]
    
    return float(np.corrcoef(values1, values2)[0, 1])

