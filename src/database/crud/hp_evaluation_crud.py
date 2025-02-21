from src.database.connection import get_connection
from src.database.object import HP_evaluation
import json

class HPEvaluationRepository:

    @staticmethod
    def create_hp_evaluation(model_idx: int, dataset_idx: int, results: list[dict]) -> HP_evaluation| None:
        query = """
        INSERT INTO hp_evaluations (model_idx, dataset_idx, results)
        VALUES (%s, %s, %s)
        RETURNING hp_evaluation_id, model_idx, dataset_idx, results, created_at;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (model_idx, dataset_idx, json.dumps(results)))
                row = cursor.fetchone()
                return HP_evaluation.from_row(row) if row else None

    @staticmethod
    def get_hp_evaluation_by_id(hp_evaluation_id: int) -> HP_evaluation | None:
        query = """
        SELECT * FROM ranks
        WHERE hp_evaluation_id = %s;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (hp_evaluation_id,))
                row = cursor.fetchone()
                return HP_evaluation.from_row(row) if row else None

    @staticmethod
    def update_results(hp_evaluation_id: int, new_results: list[dict]) -> bool:
        query = """
        UPDATE ranks
        SET results = %s
        WHERE hp_evaluation_id = %s;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (json.dumps(new_results), hp_evaluation_id))
                return cursor.rowcount > 0

    @staticmethod
    def delete_hp_evaluation(hp_evaluation_id: int) -> bool:
        query = """
        DELETE FROM ranks
        WHERE hp_evaluation_id = %s;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (hp_evaluation_id,))
                return cursor.rowcount > 0
