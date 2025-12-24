"""
Community Detection Module
Implements various community detection algorithms
"""
import networkx as nx
import community.community_louvain as community_louvain
from collections import defaultdict
from config import MIN_COMMUNITY_SIZE, MODULARITY_RESOLUTION

class CommunityDetector:
    """Detects communities in networks using various algorithms"""
    
    def __init__(self, graph):
        """
        Initialize with a network graph
        
        Args:
            graph: NetworkX graph object
        """
        self.graph = graph
        self.communities = None
        self.modularity = None
        
    def detect_louvain(self, resolution=1.0):
        """
        Detect communities using Louvain algorithm
        
        Args:
            resolution: Resolution parameter (higher = more communities)
            
        Returns:
            Dictionary mapping node to community ID
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            print("Graph is empty. Cannot detect communities.")
            return {}
        
        # Convert to undirected if needed
        if self.graph.is_directed():
            graph_undirected = self.graph.to_undirected()
        else:
            graph_undirected = self.graph
        
        # Apply Louvain algorithm
        partition = community_louvain.best_partition(
            graph_undirected, 
            resolution=resolution
        )
        
        self.communities = partition
        self.modularity = community_louvain.modularity(
            partition, 
            graph_undirected
        )
        
        print(f"Detected {len(set(partition.values()))} communities")
        print(f"Modularity: {self.modularity:.4f}")
        
        return partition
    
    def get_community_stats(self):
        """
        Get statistics about detected communities
        
        Returns:
            Dictionary with community statistics
        """
        if not self.communities:
            return {}
        
        # Group nodes by community
        community_nodes = defaultdict(list)
        for node, comm_id in self.communities.items():
            community_nodes[comm_id].append(node)
        
        # Filter small communities
        filtered_communities = {
            comm_id: nodes 
            for comm_id, nodes in community_nodes.items()
            if len(nodes) >= MIN_COMMUNITY_SIZE
        }
        
        # Calculate statistics
        community_sizes = [len(nodes) for nodes in filtered_communities.values()]
        
        stats = {
            'total_communities': len(filtered_communities),
            'total_nodes_in_communities': sum(community_sizes),
            'avg_community_size': sum(community_sizes) / len(community_sizes) if community_sizes else 0,
            'min_community_size': min(community_sizes) if community_sizes else 0,
            'max_community_size': max(community_sizes) if community_sizes else 0,
            'modularity': self.modularity,
            'communities': filtered_communities
        }
        
        return stats
    
    def get_community_members(self, community_id):
        """
        Get all members of a specific community
        
        Args:
            community_id: Community ID
            
        Returns:
            List of node IDs in the community
        """
        if not self.communities:
            return []
        
        return [
            node for node, comm_id in self.communities.items()
            if comm_id == community_id
        ]
    
    def get_node_community(self, node):
        """
        Get community ID for a specific node
        
        Args:
            node: Node ID
            
        Returns:
            Community ID or None
        """
        if not self.communities:
            return None
        
        return self.communities.get(node)
    
    def export_communities(self, filepath):
        """
        Export community assignments to file
        
        Args:
            filepath: Output file path (CSV format)
        """
        if not self.communities:
            print("No communities detected. Run detection first.")
            return
        
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['node_id', 'community_id'])
            
            for node, comm_id in self.communities.items():
                writer.writerow([node, comm_id])
        
        print(f"Communities exported to {filepath}")

