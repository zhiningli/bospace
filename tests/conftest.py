import pytest
import logging
import os
from src.database.connection import get_connection

# Set up test-specific logger
test_logger = logging.getLogger("test")

# Get the absolute path for SQL scripts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQL_SCRIPTS_DIR = os.path.join(BASE_DIR, "../../src/database/raw_sql_migration_scripts")


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create test tables before running tests and drop afterward."""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            test_logger.info("Setting up test database...")

            # Ensure the SQL scripts directory exists
            if not os.path.exists(SQL_SCRIPTS_DIR):
                raise FileNotFoundError(f"SQL scripts directory not found: {SQL_SCRIPTS_DIR}")

            # Create tables by executing each SQL script
            for script_name in [
                "create_datasets_table.sql",
                "create_models_table.sql",
                "create_hp_evaluation_table.sql",
                "create_similarity_table.sql",
                "create_results_table.sql",
                "create_scripts_table.sql",
            ]:
                script_path = os.path.join(SQL_SCRIPTS_DIR, script_name)
                if not os.path.isfile(script_path):
                    raise FileNotFoundError(f"SQL script not found: {script_path}")

                test_logger.debug(f"Executing {script_name}...")
                with open(script_path, "r") as script_file:
                    cursor.execute(script_file.read())
            
            conn.commit()
            test_logger.info("Test database setup completed successfully.")

        except Exception as e:
            conn.rollback()
            test_logger.error(f"Failed to set up test database: {e}", exc_info=True)
            raise

        finally:
            cursor.close()

    yield  # Run the tests

    # Drop tables after all tests
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            test_logger.info("Dropping test database tables...")
            drop_script_path = os.path.join(SQL_SCRIPTS_DIR, "drop_table.sql")

            if not os.path.isfile(drop_script_path):
                raise FileNotFoundError(f"Drop table script not found: {drop_script_path}")

            with open(drop_script_path, "r") as drop_file:
                cursor.execute(drop_file.read())

            conn.commit()
            test_logger.info("Test database cleanup completed successfully.")

        except Exception as e:
            conn.rollback()
            test_logger.error(f"Failed to drop test database tables: {e}", exc_info=True)
            raise

        finally:
            cursor.close()


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
