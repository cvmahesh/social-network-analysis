"""
Composite influence score calculation
"""
import numpy as np
from typing import Dict, Optional
from src.utils.metrics_utils import normalize_metrics


class CompositeScorer:
    """Calculate composite influence scores from multiple metrics"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize composite scorer
        
        Args:
            weights: Dictionary mapping metric names to weights
                    Default weights sum to 1.0
        """
        self.weights = weights or {
            'degree': 0.2,
            'pagerank': 0.3,
            'betweenness': 0.2,
            'closeness': 0.1,
            'clustering': 0.05,
            'engagement': 0.1,
            'community_leadership': 0.05
        }
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def calculate_composite_score(self,
                                 metrics: Dict[str, Dict[str, float]],
                                 normalize: bool = True) -> Dict[str, float]:
        """
        Calculate composite influence score
        
        Args:
            metrics: Dictionary mapping metric names to node scores
            normalize: Whether to normalize metrics before combining
            
        Returns:
            Dictionary mapping node to composite score
        """
        # Get all nodes
        all_nodes = set()
        for metric_dict in metrics.values():
            all_nodes.update(metric_dict.keys())
        
        # Normalize metrics if requested
        if normalize:
            normalized_metrics = {}
            for metric_name, metric_dict in metrics.items():
                normalized_metrics[metric_name] = normalize_metrics(metric_dict)
            metrics = normalized_metrics
        
        # Calculate composite score for each node
        composite_scores = {}
        for node in all_nodes:
            score = 0.0
            for metric_name, weight in self.weights.items():
                if metric_name in metrics:
                    node_score = metrics[metric_name].get(node, 0.0)
                    score += weight * node_score
            
            composite_scores[node] = score
        
        return composite_scores
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update weight configuration
        
        Args:
            new_weights: Dictionary mapping metric names to new weights
        """
        self.weights.update(new_weights)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def get_top_influencers(self,
                           composite_scores: Dict[str, float],
                           top_n: int = 10) -> list:
        """
        Get top N influencers by composite score
        
        Args:
            composite_scores: Dictionary mapping node to composite score
            top_n: Number of top influencers to return
            
        Returns:
            List of (node, score) tuples, sorted by score descending
        """
        sorted_nodes = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]

