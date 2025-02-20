from src.database.connection import get_connection


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

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            conn.commit()


if __name__ == "__main__":
    create_models_table()