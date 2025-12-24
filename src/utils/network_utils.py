"""
Network utility functions
"""
import networkx as nx
from typing import Dict, Any, Optional


def validate_graph(G: nx.Graph) -> bool:
    """
    Validate a NetworkX graph
    
    Args:
        G: NetworkX graph
        
    Returns:
        True if graph is valid
        
    Raises:
        ValueError: If graph is invalid
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("Input must be a NetworkX graph")
    
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty")
    
    return True


def get_network_stats(G: nx.Graph) -> Dict[str, Any]:
    """
    Get basic network statistics
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary with network statistics
    """
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_directed': G.is_directed(),
        'is_multigraph': G.is_multigraph(),
    }
    
    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        stats['avg_degree'] = sum(degrees.values()) / len(degrees)
        stats['max_degree'] = max(degrees.values()) if degrees else 0
        stats['min_degree'] = min(degrees.values()) if degrees else 0
    
    if not G.is_directed():
        stats['is_connected'] = nx.is_connected(G)
        if stats['is_connected']:
            stats['diameter'] = nx.diameter(G)
            stats['radius'] = nx.radius(G)
        else:
            stats['num_components'] = nx.number_connected_components(G)
            largest_cc = max(nx.connected_components(G), key=len)
            stats['largest_component_size'] = len(largest_cc)
    
    return stats


def convert_to_undirected(G: nx.Graph) -> nx.Graph:
    """
    Convert directed graph to undirected
    
    Args:
        G: NetworkX graph
        
    Returns:
        Undirected graph
    """
    if not G.is_directed():
        return G
    
    return G.to_undirected()


def get_largest_component(G: nx.Graph) -> nx.Graph:
    """
    Get largest connected component
    
    Args:
        G: NetworkX graph
        
    Returns:
        Subgraph of largest component
    """
    if G.is_directed():
        components = nx.weakly_connected_components(G)
    else:
        components = nx.connected_components(G)
    
    largest_cc = max(components, key=len)
    return G.subgraph(largest_cc)


def filter_nodes_by_degree(G: nx.Graph, min_degree: int = 1) -> nx.Graph:
    """
    Filter nodes by minimum degree
    
    Args:
        G: NetworkX graph
        min_degree: Minimum degree threshold
        
    Returns:
        Filtered graph
    """
    nodes_to_remove = [n for n, d in G.degree() if d < min_degree]
    G_filtered = G.copy()
    G_filtered.remove_nodes_from(nodes_to_remove)
    return G_filtered


def add_node_attributes_from_dict(G: nx.Graph, attributes: Dict[str, Dict[str, Any]]) -> nx.Graph:
    """
    Add node attributes from dictionary
    
    Args:
        G: NetworkX graph
        attributes: Dictionary mapping node to attribute dict
        
    Returns:
        Graph with added attributes
    """
    for node, attrs in attributes.items():
        if node in G:
            for key, value in attrs.items():
                G.nodes[node][key] = value
    return G


def add_edge_attributes_from_dict(G: nx.Graph, attributes: Dict[tuple, Dict[str, Any]]) -> nx.Graph:
    """
    Add edge attributes from dictionary
    
    Args:
        G: NetworkX graph
        attributes: Dictionary mapping (u, v) to attribute dict
        
    Returns:
        Graph with added attributes
    """
    for (u, v), attrs in attributes.items():
        if G.has_edge(u, v):
            for key, value in attrs.items():
                G[u][v][key] = value
    return G

