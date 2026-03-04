"""DuckDB connection management."""

import threading
import duckdb

_conn: duckdb.DuckDBPyConnection | None = None
_init_lock = threading.Lock()


def _get_base_connection() -> duckdb.DuckDBPyConnection:
    """Get or create the singleton DuckDB connection with views initialized."""
    global _conn
    if _conn is None:
        with _init_lock:
            if _conn is None:
                _conn = duckdb.connect(database=":memory:")
                from backend.db.schema import create_all_views
                create_all_views(_conn)
    return _conn


def get_connection() -> duckdb.DuckDBPyConnection:
    """Get a cursor for thread-safe DuckDB access.

    Each caller gets its own cursor derived from the shared connection.
    Cursors inherit all views/tables from the parent connection.
    """
    return _get_base_connection().cursor()


def reset_connection():
    """Close and reset the connection (for testing)."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
