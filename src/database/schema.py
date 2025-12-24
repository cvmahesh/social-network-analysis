"""
Database schema definitions
"""
from .database import DatabaseManager


def create_tables(db_manager: DatabaseManager):
    """
    Create all database tables
    
    Args:
        db_manager: DatabaseManager instance
    """
    # Videos table
    videos_table = """
    CREATE TABLE IF NOT EXISTS videos (
        video_id TEXT PRIMARY KEY,
        title TEXT,
        description TEXT,
        channel_id TEXT,
        channel_title TEXT,
        published_at TIMESTAMP,
        view_count INTEGER,
        like_count INTEGER,
        comment_count INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    # Comments table
    comments_table = """
    CREATE TABLE IF NOT EXISTS comments (
        comment_id TEXT PRIMARY KEY,
        video_id TEXT,
        author_id TEXT,
        author_name TEXT,
        text TEXT,
        published_at TIMESTAMP,
        like_count INTEGER,
        reply_count INTEGER,
        parent_id TEXT,
        sentiment_label TEXT,
        sentiment_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES videos(video_id)
    )
    """
    
    # Users table
    users_table = """
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        total_comments INTEGER DEFAULT 0,
        total_likes INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    # Communities table
    communities_table = """
    CREATE TABLE IF NOT EXISTS communities (
        community_id INTEGER PRIMARY KEY,
        name TEXT,
        description TEXT,
        size INTEGER,
        modularity REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    # User communities (many-to-many)
    user_communities_table = """
    CREATE TABLE IF NOT EXISTS user_communities (
        user_id TEXT,
        community_id INTEGER,
        PRIMARY KEY (user_id, community_id),
        FOREIGN KEY (user_id) REFERENCES users(user_id),
        FOREIGN KEY (community_id) REFERENCES communities(community_id)
    )
    """
    
    # Influencer metrics table
    influencer_metrics_table = """
    CREATE TABLE IF NOT EXISTS influencer_metrics (
        user_id TEXT PRIMARY KEY,
        degree_centrality REAL,
        betweenness_centrality REAL,
        pagerank REAL,
        closeness_centrality REAL,
        clustering_coefficient REAL,
        engagement_score REAL,
        community_leadership_score REAL,
        composite_influence_score REAL,
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    """
    
    # Network edges table
    network_edges_table = """
    CREATE TABLE IF NOT EXISTS network_edges (
        source_user_id TEXT,
        target_user_id TEXT,
        weight REAL,
        edge_type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (source_user_id, target_user_id),
        FOREIGN KEY (source_user_id) REFERENCES users(user_id),
        FOREIGN KEY (target_user_id) REFERENCES users(user_id)
    )
    """
    
    # Temporal snapshots table
    temporal_snapshots_table = """
    CREATE TABLE IF NOT EXISTS temporal_snapshots (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date TIMESTAMP,
        metric_name TEXT,
        metric_value REAL,
        entity_id TEXT,
        entity_type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    tables = [
        videos_table,
        comments_table,
        users_table,
        communities_table,
        user_communities_table,
        influencer_metrics_table,
        network_edges_table,
        temporal_snapshots_table
    ]
    
    for table_sql in tables:
        db_manager.execute_update(table_sql)
    
    # Create indexes
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id)",
        "CREATE INDEX IF NOT EXISTS idx_comments_author_id ON comments(author_id)",
        "CREATE INDEX IF NOT EXISTS idx_comments_published_at ON comments(published_at)",
        "CREATE INDEX IF NOT EXISTS idx_user_communities_user_id ON user_communities(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_user_communities_community_id ON user_communities(community_id)",
        "CREATE INDEX IF NOT EXISTS idx_network_edges_source ON network_edges(source_user_id)",
        "CREATE INDEX IF NOT EXISTS idx_network_edges_target ON network_edges(target_user_id)",
        "CREATE INDEX IF NOT EXISTS idx_temporal_snapshots_date ON temporal_snapshots(snapshot_date)",
        "CREATE INDEX IF NOT EXISTS idx_temporal_snapshots_entity ON temporal_snapshots(entity_id, entity_type)"
    ]
    
    for index_sql in indexes:
        try:
            db_manager.execute_update(index_sql)
        except Exception as e:
            print(f"Warning: Could not create index: {e}")


def drop_tables(db_manager: DatabaseManager):
    """
    Drop all database tables (use with caution!)
    
    Args:
        db_manager: DatabaseManager instance
    """
    tables = [
        'temporal_snapshots',
        'network_edges',
        'influencer_metrics',
        'user_communities',
        'communities',
        'comments',
        'users',
        'videos'
    ]
    
    for table in tables:
        try:
            db_manager.execute_update(f"DROP TABLE IF EXISTS {table}")
        except Exception as e:
            print(f"Warning: Could not drop table {table}: {e}")

