import pytest
import logging
from src.database.connection import get_connection

# Set up test-specific logger
test_logger = logging.getLogger("test")

# Create tables before all tests and drop them afterward
@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create test tables before running tests and drop afterward."""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            test_logger.info("Setting up test database...")

            # Create tables
            for script in [
                "create_datasets_table.sql",
                "create_models_table.sql",
                "create_hp_evaluation_table.sql",
                "create_similarity_table.sql",
                "create_results_table.sql",
                "create_scripts_table.sql",
            ]:
                with open(f"src/database/raw_sql_migration_scripts/{script}", "r") as f:
                    cursor.execute(f.read())

            conn.commit()
            test_logger.info("Test database setup completed.")

        except Exception as e:
            conn.rollback()
            test_logger.error(f"Failed to set up test database: {e}", exc_info=True)
            raise

        finally:
            cursor.close()

    yield  # Run the tests

    # Drop tables after tests
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            test_logger.info("Dropping test database tables...")
            with open("src/database/raw_sql_migration_scripts/drop_table.sql", "r") as f:
                cursor.execute(f.read())
            conn.commit()
            test_logger.info("Test database cleanup completed.")
        except Exception as e:
            conn.rollback()
            test_logger.error(f"Failed to drop test database tables: {e}", exc_info=True)
            raise
        finally:
            cursor.close()

# Transaction-based rollback for each test
@pytest.fixture(scope="function")
def db_transaction():
    """Run each test within a transaction and roll back afterward."""
    with get_connection() as conn:
        conn.autocommit = False
        try:
            test_logger.debug("Starting DB transaction for test...")
            yield conn  # Provide connection for the test
        except Exception as e:
            test_logger.error(f"Error during transaction: {e}", exc_info=True)
            raise
        finally:
            test_logger.debug("Rolling back transaction...")
            conn.rollback()
            conn.close()
