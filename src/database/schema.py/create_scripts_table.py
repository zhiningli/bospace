from src.database.connection import get_connection

def create_scripts_table():
    query = """
    CREATE TABLE IF NOT EXISTS scripts (
        script_idx SERIAL PRIMARY KEY,
        dataset_idx INT NOT NULL,
        model_idx INT NOT NULL,
        script_code TEXT NOT NULL,
        sgd_best_performing_configuration JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT fk_dataset
            FOREIGN KEY (dataset_idx)
            REFERENCE datasets (dataset_idx)
            ON DELETE CASCADE;
        CONSTRAINT fk_model
            FOREIGN KEY (model_idx)
            REFERENCE models (model_idx)
            ON DELETE CASCADE;
    );
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            conn.commit()

if __name__ == "__main__":
    create_scripts_table()