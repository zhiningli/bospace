from src.database.connection import get_connection
import logging

logger = logging.getLogger("database")

def create_datasets_table():
    """Create the 'datasets' table in PostgreSQL """
    query = """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_idx SERIAL PRIMARY KEY,
        code TEXT NOT NULL,
        input_size INTEGER NOT NULL,
        num_classes INTEGER NOT NULL,
        meta_features REAL[] NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                logging.debug("Creating dataset table")
                cursor.execute(query)
                conn.commit()
                logging.info("Dataset table created")
    except Exception as e:
        logging.error("Failed to create datasets table", exc_info=True)
if __name__ == "__main__":

    create_datasets_table()
