from src.database.connection import get_connection

def create_datasets_table():
    """Create the 'datasets' table in PostgreSQL """
    query = """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_idx SERIAL PRIMARY KEY,
        code JSONB NOT NULL,
        meta_features REAL[] NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            conn.commit()

if __name__ == "__main__":
    create_datasets_table()