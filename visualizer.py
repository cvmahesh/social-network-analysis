"""
Network Visualization Module
Creates visualizations of networks and communities
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

class NetworkVisualizer:
    """Creates visualizations for networks and communities"""
    
    def __init__(self, graph, communities=None):
        """
        Initialize visualizer
        
        Args:
            graph: NetworkX graph
            communities: Dictionary mapping node to community ID
        """
        self.graph = graph
        self.communities = communities
        
    def plot_network(self, figsize=(12, 8), node_size=50, 
                     with_labels=False, layout='spring'):
        """
        Plot network with optional community coloring
        
        Args:
            figsize: Figure size tuple
            node_size: Size of nodes
            with_labels: Whether to show node labels
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            print("Graph is empty. Cannot visualize.")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Color nodes by community if available
        if self.communities:
            # Get unique communities
            unique_communities = set(self.communities.values())
            n_communities = len(unique_communities)
            
            # Generate colors
            colors = plt.cm.tab20(np.linspace(0, 1, n_communities))
            node_colors = [
                colors[list(unique_communities).index(self.communities.get(node, -1)) % n_communities]
                for node in self.graph.nodes()
            ]
        else:
            node_colors = 'lightblue'
        
        # Draw network
        nx.draw_networkx_nodes(
            self.graph, pos, 
            node_color=node_colors,
            node_size=node_size,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            self.graph, pos,
            alpha=0.3,
            width=0.5,
            ax=ax
        )
        
        if with_labels:
            nx.draw_networkx_labels(
                self.graph, pos,
                font_size=8,
                ax=ax
            )
        
        ax.set_title("Network Visualization with Communities", fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_community_size_distribution(self, figsize=(10, 6)):
        """
        Plot distribution of community sizes
        
        Args:
            figsize: Figure size tuple
        """
        if not self.communities:
            print("No communities detected. Run detection first.")
            return
        
        # Count community sizes
        from collections import Counter
        community_sizes = Counter(self.communities.values())
        sizes = list(community_sizes.values())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(sizes, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Community Size', fontsize=12)
        ax.set_ylabel('Number of Communities', fontsize=12)
        ax.set_title('Community Size Distribution', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_degree_distribution(self, figsize=(10, 6)):
        """
        Plot degree distribution of the network
        
        Args:
            figsize: Figure size tuple
        """
        if not self.graph:
            print("Graph is empty. Cannot plot degree distribution.")
            return
        
        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.hist(degree_values, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Degree', fontsize=12)
        ax.set_ylabel('Number of Nodes', fontsize=12)
        ax.set_title('Degree Distribution', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_plot(self, fig, filepath, dpi=300):
        """
        Save plot to file
        
        Args:
            fig: Matplotlib figure object
            filepath: Output file path
            dpi: Resolution
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {filepath}")

