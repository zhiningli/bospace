import json
from src.database.connection import get_connection

def insert_dataset(code, meta_features):
    """Insert a new data record"""
    query = """
    INSERT INTO datasets (code, meta_features)
    VALUES (%s, %s)
    RETURNING dataset_idx;
    """

    with get_connection as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (json.dumps(code), meta_features))
            dataset_idx = cursor.fetchone()[0]
            conn.commit()