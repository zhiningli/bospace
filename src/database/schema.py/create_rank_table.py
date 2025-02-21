from src.database.connection import get_connection

def create_rank_table():
    # Create ENUM type if not exists
    query_enum = """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'rank_object_enum') THEN
            CREATE TYPE rank_object_enum AS ENUM ('model', 'dataset');
        END IF;
    END $$;
    """

    # Create the ranks table with polymorphic foreign keys for both objects
    query_table = """
    CREATE TABLE IF NOT EXISTS ranks (
        ranks_idx SERIAL PRIMARY KEY,
        object_type rank_object_enum NOT NULL,
        object_1_idx INT NOT NULL,
        object_1 REAL[] NOT NULL,

        object_2_idx INT NOT NULL,
        object_2 REAL[] NOT NULL,

        rank REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        -- Check constraints for polymorphic relationships
        CONSTRAINT valid_object_1
        CHECK (
            (object_type = 'dataset' AND EXISTS (SELECT 1 FROM datasets WHERE dataset_idx = object_1_idx))
            OR
            (object_type = 'model' AND EXISTS (SELECT 1 FROM models WHERE model_idx = object_1_idx))
        ),

        CONSTRAINT valid_object_2
        CHECK (
            (object_type = 'dataset' AND EXISTS (SELECT 1 FROM datasets WHERE dataset_idx = object_2_idx))
            OR
            (object_type = 'model' AND EXISTS (SELECT 1 FROM models WHERE model_idx = object_2_idx))
        )
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
    create_rank_table()
