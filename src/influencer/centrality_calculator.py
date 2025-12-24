"""
Centrality metrics calculation for influencer identification
"""
import networkx as nx
from typing import Dict, Optional, List
import warnings


class CentralityCalculator:
    """Calculate various centrality metrics for network nodes"""
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize centrality calculator
        
        Args:
            graph: NetworkX graph
        """
        self.graph = graph
        self._cache = {}
    
    def calculate_degree_centrality(self, normalized: bool = True) -> Dict[str, float]:
        """
        Calculate degree centrality
        
        Args:
            normalized: Whether to normalize (default: True)
            
        Returns:
            Dictionary mapping node to degree centrality score
        """
        if normalized:
            return nx.degree_centrality(self.graph)
        else:
            return dict(self.graph.degree())
    
    def calculate_betweenness_centrality(self, 
                                        k: Optional[int] = None,
                                        normalized: bool = True) -> Dict[str, float]:
        """
        Calculate betweenness centrality
        
        Args:
            k: Number of nodes to sample for approximation (None for exact)
            normalized: Whether to normalize
            
        Returns:
            Dictionary mapping node to betweenness centrality score
        """
        if k and self.graph.number_of_nodes() > k:
            # Use approximate algorithm for large networks
            sample_nodes = list(self.graph.nodes())[:k]
            return nx.betweenness_centrality(self.graph, k=sample_nodes, normalized=normalized)
        else:
            return nx.betweenness_centrality(self.graph, normalized=normalized)
    
    def calculate_closeness_centrality(self, 
                                     use_harmonic: bool = False) -> Dict[str, float]:
        """
        Calculate closeness centrality
        
        Args:
            use_harmonic: Use harmonic closeness (works for disconnected graphs)
            
        Returns:
            Dictionary mapping node to closeness centrality score
        """
        if use_harmonic:
            return nx.harmonic_centrality(self.graph)
        else:
            # Check if graph is connected
            if not self.graph.is_directed():
                if not nx.is_connected(self.graph):
                    warnings.warn(
                        "Graph is not connected. Using harmonic closeness.",
                        UserWarning
                    )
                    return nx.harmonic_centrality(self.graph)
            
            return nx.closeness_centrality(self.graph)
    
    def calculate_pagerank(self, 
                          alpha: float = 0.85,
                          max_iter: int = 100,
                          tol: float = 1e-06) -> Dict[str, float]:
        """
        Calculate PageRank
        
        Args:
            alpha: Damping factor (default: 0.85)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Dictionary mapping node to PageRank score
        """
        return nx.pagerank(self.graph, alpha=alpha, max_iter=max_iter, tol=tol)
    
    def calculate_eigenvector_centrality(self,
                                        max_iter: int = 100,
                                        tol: float = 1e-06) -> Dict[str, float]:
        """
        Calculate eigenvector centrality
        
        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Dictionary mapping node to eigenvector centrality score
        """
        try:
            return nx.eigenvector_centrality(self.graph, max_iter=max_iter, tol=tol)
        except nx.PowerIterationFailedConvergence:
            warnings.warn("Eigenvector centrality failed to converge", UserWarning)
            return {node: 0.0 for node in self.graph.nodes()}
    
    def calculate_clustering_coefficient(self) -> Dict[str, float]:
        """
        Calculate local clustering coefficient
        
        Returns:
            Dictionary mapping node to clustering coefficient
        """
        return nx.clustering(self.graph)
    
    def calculate_all_centralities(self,
                                   include_expensive: bool = True,
                                   approximate_large: bool = True,
                                   sample_size: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        Calculate all centrality metrics
        
        Args:
            include_expensive: Include expensive metrics (betweenness, closeness)
            approximate_large: Use approximation for large networks
            sample_size: Sample size for approximation
            
        Returns:
            Dictionary mapping metric name to node scores
        """
        results = {}
        
        # Always calculate these (fast)
        results['degree'] = self.calculate_degree_centrality()
        results['pagerank'] = self.calculate_pagerank()
        results['clustering'] = self.calculate_clustering_coefficient()
        
        if include_expensive:
            # Use approximation for large networks
            if approximate_large and self.graph.number_of_nodes() > sample_size:
                results['betweenness'] = self.calculate_betweenness_centrality(k=sample_size)
            else:
                results['betweenness'] = self.calculate_betweenness_centrality()
            
            # Closeness (use harmonic for disconnected graphs)
            results['closeness'] = self.calculate_closeness_centrality(use_harmonic=True)
        
        # Eigenvector (can be slow)
        try:
            results['eigenvector'] = self.calculate_eigenvector_centrality()
        except Exception as e:
            warnings.warn(f"Could not calculate eigenvector centrality: {e}", UserWarning)
        
        return results
    
    def get_top_nodes(self, 
                     metric: str,
                     top_n: int = 10) -> List[tuple]:
        """
        Get top N nodes by a centrality metric
        
        Args:
            metric: Metric name ('degree', 'pagerank', etc.)
            top_n: Number of top nodes to return
            
        Returns:
            List of (node, score) tuples, sorted by score descending
        """
        if metric == 'degree':
            scores = self.calculate_degree_centrality()
        elif metric == 'betweenness':
            scores = self.calculate_betweenness_centrality()
        elif metric == 'closeness':
            scores = self.calculate_closeness_centrality()
        elif metric == 'pagerank':
            scores = self.calculate_pagerank()
        elif metric == 'eigenvector':
            scores = self.calculate_eigenvector_centrality()
        elif metric == 'clustering':
            scores = self.calculate_clustering_coefficient()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]

