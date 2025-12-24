"""
Database operations for storing YouTube data and analysis results
"""
import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import os

# Try to import config, use defaults if not available
try:
    from config import DB_NAME, DB_HOST, DB_PORT, DB_USER, DB_PASSWORD
except ImportError:
    DB_NAME = 'youtube_communities'
    DB_HOST = 'localhost'
    DB_PORT = '5432'
    DB_USER = ''
    DB_PASSWORD = ''


class DatabaseManager:
    """Database manager for SQLite/PostgreSQL operations"""
    
    def __init__(self, db_path: Optional[str] = None, db_type: str = 'sqlite'):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database or connection string
            db_type: Database type ('sqlite' or 'postgresql')
        """
        self.db_type = db_type
        self.db_path = db_path or f"{DB_NAME}.db"
        self.connection = None
        
        if db_type == 'sqlite':
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Get database connection (context manager)"""
        if self.db_type == 'sqlite':
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
        else:
            # PostgreSQL connection would go here
            import psycopg2
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
        
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            return results
    
    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            return cursor.rowcount
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """
        Insert DataFrame into database table
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            if_exists: What to do if table exists ('fail', 'replace', 'append')
        """
        with self.get_connection() as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    
    def read_dataframe(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Read data into DataFrame
        
        Args:
            query: SQL SELECT query
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        with self.get_connection() as conn:
            if params:
                return pd.read_sql_query(query, conn, params=params)
            else:
                return pd.read_sql_query(query, conn)
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        if self.db_type == 'sqlite':
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            results = self.execute_query(query, (table_name,))
            return len(results) > 0
        else:
            # PostgreSQL check
            query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
            """
            results = self.execute_query(query, (table_name,))
            return results[0]['exists'] if results else False
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema information"""
        if self.db_type == 'sqlite':
            query = f"PRAGMA table_info({table_name})"
            return self.execute_query(query)
        else:
            query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s
            """
            return self.execute_query(query, (table_name,))

