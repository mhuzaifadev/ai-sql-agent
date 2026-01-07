"""
Database connection module for PostgreSQL.

Provides connection pooling, query execution, and schema retrieval.
"""

import os
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()


class DatabaseConnection:
    """Manages PostgreSQL database connections with connection pooling."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 5,
    ):
        """
        Initialize database connection manager.

        Args:
            host: Database host (defaults to DB_HOST env var)
            port: Database port (defaults to DB_PORT env var)
            database: Database name (defaults to DB_NAME env var)
            user: Database user (defaults to DB_USER env var)
            password: Database password (defaults to DB_PASSWORD env var)
            min_connections: Minimum pool size
            max_connections: Maximum pool size
        """
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", "5432"))
        self.database = database or os.getenv("DB_NAME")
        self.user = user or os.getenv("DB_USER")
        self.password = password or os.getenv("DB_PASSWORD")

        if not all([self.database, self.user, self.password]):
            raise ValueError(
                "Database credentials not provided. Set DB_NAME, DB_USER, and DB_PASSWORD "
                "in environment variables or pass them as arguments."
            )

        self.connection_pool: Optional[pool.ThreadedConnectionPool] = None
        self.min_connections = min_connections
        self.max_connections = max_connections

    def connect(self) -> None:
        """Establish connection pool."""
        try:
            self.connection_pool = pool.ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
        except psycopg2.Error as e:
            raise ConnectionError(f"Failed to create connection pool: {e}")

    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a connection from the pool.

        Yields:
            Database connection object

        Raises:
            ConnectionError: If pool is not initialized or connection fails
        """
        if self.connection_pool is None:
            self.connect()

        conn = None
        try:
            conn = self.connection_pool.getconn()
            if conn is None:
                raise ConnectionError("Failed to get connection from pool")
            # Set read-only mode
            conn.set_session(readonly=True, autocommit=True)
            yield conn
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def execute_query(
        self, sql_query: str, params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.

        Args:
            sql_query: SQL SELECT query to execute
            params: Optional query parameters for parameterized queries

        Returns:
            List of dictionaries representing query results

        Raises:
            ValueError: If query is not a SELECT statement
            psycopg2.Error: If query execution fails
        """
        # Basic validation - ensure it's a SELECT query
        query_upper = sql_query.strip().upper()
        if not query_upper.startswith("SELECT"):
            raise ValueError(
                "Only SELECT queries are allowed. "
                f"Query starts with: {query_upper.split()[0] if query_upper.split() else 'empty'}"
            )

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                try:
                    if params:
                        cursor.execute(sql_query, params)
                    else:
                        cursor.execute(sql_query)
                    results = cursor.fetchall()
                    # Convert RealDictRow to regular dict
                    return [dict(row) for row in results]
                except psycopg2.Error as e:
                    raise psycopg2.Error(f"Query execution failed: {e}")

    def get_table_schemas(
        self, schema_name: str = "public"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve database schema metadata for all tables.

        Args:
            schema_name: Schema name to query (default: 'public')

        Returns:
            List of dictionaries containing table schema information
        """
        query = """
        SELECT 
            t.table_name,
            t.table_type,
            json_agg(
                json_build_object(
                    'column_name', c.column_name,
                    'data_type', c.data_type,
                    'is_nullable', c.is_nullable,
                    'column_default', c.column_default
                ) ORDER BY c.ordinal_position
            ) as columns
        FROM information_schema.tables t
        LEFT JOIN information_schema.columns c 
            ON t.table_schema = c.table_schema 
            AND t.table_name = c.table_name
        WHERE t.table_schema = %s
            AND t.table_type = 'BASE TABLE'
        GROUP BY t.table_name, t.table_type
        ORDER BY t.table_name;
        """

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (schema_name,))
                results = cursor.fetchall()
                return [dict(row) for row in results]

    def get_table_info(self, table_name: str, schema_name: str = "public") -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific table.

        Args:
            table_name: Name of the table
            schema_name: Schema name (default: 'public')

        Returns:
            Dictionary with table information or None if table doesn't exist
        """
        schemas = self.get_table_schemas(schema_name)
        for schema in schemas:
            if schema["table_name"] == table_name:
                return schema
        return None

    def test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close all connections in the pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.connection_pool = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

