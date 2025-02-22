from src.database.connection import get_connection
import logging

logger=logging.getLogger("database")
def drop_table(table_name: str):

    query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                logger.debug(f"Dropping table {table_name}")
                cursor.execute(query)
                conn.commit()
                logger.info(f"Table '{table_name}' dropped successfully!")

    except Exception as e:
        logger.error(f"Failed to drop table '{table_name}': {e}")

if __name__ == "__main__":
    drop_table("ranks")