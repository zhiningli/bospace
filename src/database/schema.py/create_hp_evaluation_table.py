from src.database.connection import get_connection

import logging

logger = logging.getLogger("database")
def create_hp_evaluation_table():

    query = """
    CREATE TABLE IF NOT EXISTS hp_evaluations (
        hp_evaluation_id SERIAL PRIMARY KEY,
        model_idx INTEGER NOT NULL,
        dataset_idx INTEGER NOT NULL,
        results JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (model_idx) REFERENCES models(model_idx) ON DELETE CASCADE,
        FOREIGN KEY (dataset_idx) REFERENCES datasets(dataset_idx) ON DELETE CASCADE
    );
    """
    try: 
        with get_connection() as conn:
            with conn.cursor() as cursor:
                logger.debug("Creating hp_evaluation_table")
                cursor.execute(query)
                conn.commit()
                logger.info("hp_evaluation_table created")
    except Exception as e:
        logger.error("Error creating hp evaluation table", exc_info=True)

if __name__ == "__main__":

    create_hp_evaluation_table()


