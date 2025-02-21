from src.database.connection import get_connection
from src.database.object.result import Result
import json

class ResultRepository:
    """CRUD operations for the 'results' table."""

    @staticmethod
    def create_result(script_idx: int, model_idx: int, dataset_idx: int, result_type: str, sgd_config: dict) -> Result | None:
        """Create a new result entry."""
        query = """
        INSERT INTO results (script_idx, model_idx, dataset_idx, result_type, sgd_best_performing_configuration)
        VALUES (%s, %s, %s, %s::result_type_enum, %s)
        RETURNING result_id, script_idx, model_idx, dataset_idx, result_type, sgd_best_performing_configuration, created_at;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (script_idx, model_idx, dataset_idx, result_type, json.dumps(sgd_config)))
                row = cursor.fetchone()
                return Result.from_row(row) if row else None

    @staticmethod
    def get_result(result_id: int) -> Result | None:
        """Retrieve a specific result by its ID."""
        query = """
        SELECT * FROM results
        WHERE result_id = %s;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (result_id,))
                row = cursor.fetchone()
                return Result.from_row(row) if row else None

    @staticmethod
    def get_results_by_script(script_idx: int) -> list[Result]:
        """Retrieve all results for a specific script."""
        query = """
        SELECT * FROM results
        WHERE script_idx = %s;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (script_idx,))
                rows = cursor.fetchall()
                return [Result.from_row(row) for row in rows]

    @staticmethod
    def update_sgd_config(script_idx: int, result_type: str, new_config: dict) -> bool:
        """Update the SGD best performing configuration for a specific script and result type."""
        query = """
        UPDATE results
        SET sgd_best_performing_configuration = %s
        WHERE script_idx = %s AND result_type = %s::result_type_enum;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (json.dumps(new_config), script_idx, result_type))
                return cursor.rowcount > 0


    @staticmethod
    def delete_result(result_id: int) -> bool:
        """Delete a result by its ID."""
        query = """
        DELETE FROM results
        WHERE result_id = %s;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (result_id,))
                return cursor.rowcount > 0
