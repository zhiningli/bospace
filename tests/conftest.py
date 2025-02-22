import pytest
from src.database.connection import get_connection

# Create tables before all tests
@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create test tables before running tests and drop afterward."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Create tables (use your existing SQL scripts)
        cursor.execute(open("src/database/raw_sql_migration_scripts/create_datasets_table.sql").read())
        cursor.execute(open("src/database/raw_sql_migration_scripts/create_models_table.sql").read())
        cursor.execute(open("src/database/raw_sql_migration_scripts/create_hp_evaluation_table.sql").read())
        cursor.execute(open("src/database/raw_sql_migration_scripts/create_similarity_table.sql").read())
        cursor.execute(open("src/database/raw_sql_migration_scripts/create_results_table.sql").read())
        cursor.execute(open("src/database/raw_sql_migration_scripts/create_scripts_table.sql").read())

        conn.commit()
        cursor.close()

    yield  # Run tests

    # Drop tables after tests
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(open("src/database/raw_sql_migration_scripts/drop_table.sql").read())
        conn.commit()
        cursor.close()

