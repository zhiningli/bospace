from src.database.connection import get_connection


def create_ranks_table():

    query = """
    CREATE TABLE IF NOT EXISTS ranks (
        rank_id SERIAL PRIMARY KEY,
        model_idx INTEGER NOT NULL,
        datset_idx INTERGET NOT NULL,
        results JSONB[] NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (model_idx) REFERENCES models(model_idx) ON DELETE CASCADE,
        FOREIGN KEY (dataset_idx) REFERENCES datasets(datset_idx) ON DELETE CASCADE
    );
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            conn.commit()

if __name__ == "__main__":
    create_ranks_table()
