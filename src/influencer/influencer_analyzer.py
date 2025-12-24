"""
Main influencer identification system
"""
import networkx as nx
import pandas as pd
from typing import Dict, List, Optional
from .centrality_calculator import CentralityCalculator
from .composite_scorer import CompositeScorer
import community.community_louvain as community_louvain


class InfluencerAnalyzer:
    """Complete influencer identification system"""
    
    def __init__(self, graph: nx.Graph, communities: Optional[Dict[str, int]] = None):
        """
        Initialize influencer analyzer
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping node to community ID (optional)
        """
        self.graph = graph
        self.communities = communities
        self.centrality_calc = CentralityCalculator(graph)
        self.composite_scorer = CompositeScorer()
        self.metrics = {}
        self.composite_scores = {}
    
    def detect_communities(self, resolution: float = 1.0):
        """
        Detect communities using Louvain algorithm
        
        Args:
            resolution: Resolution parameter for Louvain
        """
        if self.graph.is_directed():
            graph_undirected = self.graph.to_undirected()
        else:
            graph_undirected = self.graph
        
        self.communities = community_louvain.best_partition(
            graph_undirected,
            resolution=resolution
        )
        return self.communities
    
    def calculate_centrality_metrics(self, include_expensive: bool = True):
        """Calculate all centrality metrics"""
        self.metrics.update(
            self.centrality_calc.calculate_all_centralities(include_expensive=include_expensive)
        )
        return self.metrics
    
    def calculate_community_leadership(self) -> Dict[str, float]:
        """
        Calculate community leadership scores
        
        Returns:
            Dictionary mapping node to community leadership score
        """
        if not self.communities:
            self.detect_communities()
        
        leadership_scores = {}
        
        # Group nodes by community
        comm_nodes = {}
        for node, comm_id in self.communities.items():
            if comm_id not in comm_nodes:
                comm_nodes[comm_id] = []
            comm_nodes[comm_id].append(node)
        
        # Calculate leadership within each community
        for comm_id, nodes in comm_nodes.items():
            G_sub = self.graph.subgraph(nodes)
            
            if G_sub.number_of_nodes() > 0:
                # Calculate PageRank within community
                pr_sub = nx.pagerank(G_sub)
                
                # Normalize by max value in community
                max_pr = max(pr_sub.values()) if pr_sub.values() else 1.0
                
                for node in nodes:
                    if node in pr_sub and max_pr > 0:
                        leadership_scores[node] = pr_sub[node] / max_pr
                    else:
                        leadership_scores[node] = 0.0
        
        self.metrics['community_leadership'] = leadership_scores
        return leadership_scores
    
    def calculate_engagement_scores(self, 
                                    comments_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate engagement scores from comments data
        
        Args:
            comments_data: DataFrame with columns: author_id, replies, likes
            
        Returns:
            Dictionary mapping user_id to engagement score
        """
        if comments_data is None:
            return {}
        
        engagement_scores = {}
        
        if 'author_id' in comments_data.columns:
            user_stats = comments_data.groupby('author_id').agg({
                'replies': 'sum' if 'replies' in comments_data.columns else 'count',
                'likes': 'sum' if 'likes' in comments_data.columns else 'count'
            }).reset_index()
            
            for _, row in user_stats.iterrows():
                user_id = row['author_id']
                total_replies = row.get('replies', 0)
                total_likes = row.get('likes', 0)
                comment_count = len(comments_data[comments_data['author_id'] == user_id])
                
                # Normalized engagement score
                engagement_scores[user_id] = (total_replies + total_likes) / max(comment_count, 1)
        
        self.metrics['engagement'] = engagement_scores
        return engagement_scores
    
    def calculate_composite_score(self, 
                                  include_engagement: bool = False,
                                  include_community: bool = True) -> Dict[str, float]:
        """
        Calculate composite influence score
        
        Args:
            include_engagement: Whether to include engagement metrics
            include_community: Whether to include community leadership
            
        Returns:
            Dictionary mapping node to composite score
        """
        # Prepare metrics dictionary
        metrics_to_combine = {}
        
        # Add centrality metrics
        if 'degree' in self.metrics:
            metrics_to_combine['degree'] = self.metrics['degree']
        if 'pagerank' in self.metrics:
            metrics_to_combine['pagerank'] = self.metrics['pagerank']
        if 'betweenness' in self.metrics:
            metrics_to_combine['betweenness'] = self.metrics['betweenness']
        if 'closeness' in self.metrics:
            metrics_to_combine['closeness'] = self.metrics['closeness']
        
        # Add optional metrics
        if include_community and 'community_leadership' in self.metrics:
            metrics_to_combine['community_leadership'] = self.metrics['community_leadership']
        
        if include_engagement and 'engagement' in self.metrics:
            metrics_to_combine['engagement'] = self.metrics['engagement']
        
        # Calculate composite score
        self.composite_scores = self.composite_scorer.calculate_composite_score(metrics_to_combine)
        
        return self.composite_scores
    
    def get_top_influencers(self, n: int = 10, metric: str = 'composite') -> List[tuple]:
        """
        Get top N influencers
        
        Args:
            n: Number of top influencers
            metric: Metric to use ('composite', 'degree', 'pagerank', etc.)
            
        Returns:
            List of (node, score) tuples
        """
        if metric == 'composite':
            if not self.composite_scores:
                self.calculate_composite_score()
            scores = self.composite_scores
        else:
            scores = self.metrics.get(metric, {})
        
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:n]
    
    def export_results(self, filepath: str = 'influencer_results.csv'):
        """
        Export influencer analysis results to CSV
        
        Args:
            filepath: Output file path
        """
        results = []
        
        for node in self.graph.nodes():
            result = {'node_id': node}
            
            # Add all metrics
            for metric_name, metric_dict in self.metrics.items():
                result[metric_name] = metric_dict.get(node, 0.0)
            
            # Add composite score
            result['composite_score'] = self.composite_scores.get(node, 0.0)
            
            # Add community
            if self.communities:
                result['community_id'] = self.communities.get(node, -1)
            
            results.append(result)
        
        df = pd.DataFrame(results)
        df = df.sort_values('composite_score', ascending=False)
        df.to_csv(filepath, index=False)
        
        return df

