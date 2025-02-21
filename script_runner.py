from src.database.connection import get_connection

def create_similarity_table():
    # Create ENUM type if not exists
    query_enum = """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'similarity_object_enum') THEN
            CREATE TYPE similarity_object_enum AS ENUM ('model', 'dataset');
        END IF;
    END $$;
    """

    # Create the similarities table with polymorphic foreign keys for both objects
    query_table = """
    CREATE TABLE IF NOT EXISTS similarities (
        similarity_idx SERIAL PRIMARY KEY,
        object_type similarity_object_enum NOT NULL, -- 'dataset' or 'model' for both objects
        object_1_idx INT NOT NULL,
        object_1 REAL[] NOT NULL,

        object_2_idx INT NOT NULL,
        object_2 REAL[] NOT NULL,

        similarity REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        -- Foreign keys based on object_type
        CONSTRAINT fk_object_1_dataset
        FOREIGN KEY (object_1_idx)
        REFERENCES datasets(dataset_idx)
        DEFERRABLE INITIALLY DEFERRED,

        CONSTRAINT fk_object_1_model
        FOREIGN KEY (object_1_idx)
        REFERENCES models(model_idx)
        DEFERRABLE INITIALLY DEFERRED,

        CONSTRAINT fk_object_2_dataset
        FOREIGN KEY (object_2_idx)
        REFERENCES datasets(dataset_idx)
        DEFERRABLE INITIALLY DEFERRED,

        CONSTRAINT fk_object_2_model
        FOREIGN KEY (object_2_idx)
        REFERENCES models(model_idx)
        DEFERRABLE INITIALLY DEFERRED
    );
    """


    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # Create ENUM type
                cursor.execute(query_enum)

                # Create Ranks Table
                cursor.execute(query_table)

                conn.commit()

        print(" Ranks table created successfully with polymorphic keys for both objects!")

    except Exception as e:
        print(f"Failed to create ranks table: {e}")

if __name__ == "__main__":
    create_similarity_table()
