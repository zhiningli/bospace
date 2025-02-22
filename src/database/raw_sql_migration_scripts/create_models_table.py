from src.database.connection import get_connection

import logging

logger = logging.getLogger("database")
def create_models_table():
    """Crete models table in postgresql"""

    query = """
    CREATE TABLE IF NOT EXISTS models (
    model_idx SERIAL PRIMARY KEY,
    code TEXT NOT NULL,
    feature_vector REAL[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                logger.debug("creating models table")
                cursor.execute(query)
                conn.commit()
                logger.info("models table created")
        
    except Exception as e:
        logger.error("Error creating models table")

if __name__ == "__main__":

    create_models_table()
