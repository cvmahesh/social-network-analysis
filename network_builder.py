"""
Network Construction Module
Builds user-user networks from YouTube comment data
"""
import networkx as nx
from collections import defaultdict
from config import MIN_COMMENTS_PER_USER

class NetworkBuilder:
    """Builds and manages network graphs from YouTube data"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.user_videos = defaultdict(set)  # Track which videos each user commented on
        
    def build_user_network_from_comments(self, comments):
        """
        Build a user-user network from comments
        
        Network structure:
        - Nodes: Users (commenters)
        - Edges: Users who commented on the same videos
        - Edge Weight: Number of shared videos (Jaccard similarity)
        
        Args:
            comments: List of comment dictionaries
        """
        # Reset graph
        self.graph = nx.Graph()
        self.user_videos = defaultdict(set)
        
        # Track user activity
        user_comment_count = defaultdict(int)
        
        # First pass: collect user-video relationships
        for comment in comments:
            author_id = comment.get('author_id') or comment.get('author', '')
            video_id = comment.get('video_id', '')
            
            if author_id and video_id:
                self.user_videos[author_id].add(video_id)
                user_comment_count[author_id] += 1
        
        # Filter users with minimum comments
        active_users = {
            user for user, count in user_comment_count.items()
            if count >= MIN_COMMENTS_PER_USER
        }
        
        # Add nodes
        for user in active_users:
            self.graph.add_node(user, comment_count=user_comment_count[user])
        
        # Add edges based on shared videos
        user_list = list(active_users)
        for i, user1 in enumerate(user_list):
            for user2 in user_list[i+1:]:
                shared_videos = self.user_videos[user1] & self.user_videos[user2]
                
                if shared_videos:
                    # Calculate Jaccard similarity
                    union_videos = self.user_videos[user1] | self.user_videos[user2]
                    jaccard = len(shared_videos) / len(union_videos) if union_videos else 0
                    
                    # Add edge with weight
                    self.graph.add_edge(
                        user1, 
                        user2, 
                        weight=len(shared_videos),
                        jaccard=jaccard
                    )
        
        print(f"Network built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def build_reply_network(self, comments):
        """
        Build a network based on reply interactions
        
        Network structure:
        - Nodes: Users
        - Edges: User A replied to User B
        - Edge Weight: Number of replies
        
        Args:
            comments: List of comment dictionaries with parent_id field
        """
        self.graph = nx.DiGraph()  # Directed graph for replies
        
        # Track reply relationships
        reply_edges = defaultdict(int)
        
        for comment in comments:
            author_id = comment.get('author_id') or comment.get('author', '')
            parent_id = comment.get('parent_id')
            
            if parent_id and author_id:
                # Find parent comment author
                for parent_comment in comments:
                    if parent_comment.get('comment_id') == parent_id:
                        parent_author = parent_comment.get('author_id') or parent_comment.get('author', '')
                        if parent_author and parent_author != author_id:
                            reply_edges[(author_id, parent_author)] += 1
                        break
        
        # Build graph
        for (source, target), weight in reply_edges.items():
            if self.graph.has_edge(source, target):
                self.graph[source][target]['weight'] += weight
            else:
                self.graph.add_edge(source, target, weight=weight)
        
        print(f"Reply network built: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def get_network_stats(self):
        """Get basic network statistics"""
        if not self.graph:
            return {}
        
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph) if not self.graph.is_directed() else False,
        }
        
        if self.graph.number_of_nodes() > 0:
            degrees = dict(self.graph.degree())
            stats['avg_degree'] = sum(degrees.values()) / len(degrees)
            stats['max_degree'] = max(degrees.values())
        
        return stats
    
    def save_network(self, filepath, format='gexf'):
        """
        Save network to file
        
        Args:
            filepath: Output file path
            format: File format ('gexf', 'graphml', 'gml')
        """
        if format == 'gexf':
            nx.write_gexf(self.graph, filepath)
        elif format == 'graphml':
            nx.write_graphml(self.graph, filepath)
        elif format == 'gml':
            nx.write_gml(self.graph, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Network saved to {filepath}")

