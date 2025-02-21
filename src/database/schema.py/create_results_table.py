from src.database.connection import get_connection

def create_results_table():
    query_enum = """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'result_type_enum') THEN
            CREATE TYPE result_type_enum AS ENUM (
                'unconstrained',
                'constrained',
                'unseen_constrained',
                'improved_constrained',
                'improved_unseen_constrained'
            );
        END IF;
    END $$;
    """

    query_table = """
    CREATE TABLE IF NOT EXISTS results (
        result_id SERIAL PRIMARY KEY,
        script_idx INTEGER NOT NULL,
        model_idx INTEGER NOT NULL,
        dataset_idx INTEGER NOT NULL,
        result_type result_type_enum NOT NULL,
        sgd_best_performing_configuration JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT fk_script FOREIGN KEY (script_idx) REFERENCES scripts(script_idx) ON DELETE CASCADE,
        CONSTRAINT fk_model FOREIGN KEY (model_idx) REFERENCES models(model_idx) ON DELETE CASCADE,
        CONSTRAINT fk_dataset FOREIGN KEY (dataset_idx) REFERENCES datasets(dataset_idx) ON DELETE CASCADE
    );
    """

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query_enum)
            cursor.execute(query_table)
            conn.commit()
    
if __name__ == "__main__":
    create_results_table()