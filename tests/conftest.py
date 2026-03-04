"""
Pytest bootstrap for NeuralRP test suite.

Ensures database schema exists before tests import modules that query tables.
"""

from app.database import initialize_database_runtime


def pytest_sessionstart(session):
    """Initialize DB runtime once for the full test session."""
    ok = initialize_database_runtime(force=True)
    if not ok:
        raise RuntimeError("Failed to initialize database runtime for tests")

