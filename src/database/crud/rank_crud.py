from src.database.connection import get_connection
from src.database.object import Rank
import json

class RankRepository:

    @staticmethod
    def create_rank(model_idx: int, dataset_idx: int, results: list[dict]) -> Rank | None:
        query = """
        INSERT INTO ranks (model_idx, dataset_idx, results)
        VALUES (%s, %s, %s)
        RETURNING rank_id, model_idx, dataset_idx, results, created_at;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (model_idx, dataset_idx, json.dumps(results)))
                row = cursor.fetchone()
                return Rank.from_row(row) if row else None

    @staticmethod
    def get_rank(rank_id: int) -> Rank | None:
        query = """
        SELECT * FROM ranks
        WHERE rank_id = %s;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (rank_id,))
                row = cursor.fetchone()
                return Rank.from_row(row) if row else None

    @staticmethod
    def update_results(rank_id: int, new_results: list[dict]) -> bool:
        query = """
        UPDATE ranks
        SET results = %s
        WHERE rank_id = %s;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (json.dumps(new_results), rank_id))
                return cursor.rowcount > 0

    @staticmethod
    def delete_rank(rank_id: int) -> bool:
        query = """
        DELETE FROM ranks
        WHERE rank_id = %s;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (rank_id,))
                return cursor.rowcount > 0
