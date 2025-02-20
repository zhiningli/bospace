from src.database.connection import get_connection

def drop_table(table_name: str):

    query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                conn.commit()
                print(f"Table '{table_name}' dropped successfully!")
    except Exception as e:
        print(f"Failed to drop table '{table_name}': {e}")

if __name__ == "__main__":
    drop_table("datasets")