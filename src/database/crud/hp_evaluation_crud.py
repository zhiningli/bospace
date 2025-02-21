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
        SELECT * FROM hp_evaluations
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
        UPDATE hp_evalutaions
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
        DELETE FROM hp_evaluations
        WHERE hp_evaluation_id = %s;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (hp_evaluation_id,))
                return cursor.rowcount > 0

    @staticmethod
    def exists_hp_evaluation(model_idx: int, dataset_idx: int) -> bool:
        """Check if a model_idx and dataset_idx pair exists in the repository."""
        query = """
        SELECT EXISTS(
            SELECT 1 
            FROM hp_evaluations
            WHERE model_idx = %s 
            AND dataset_idx = %s
        );
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (model_idx, dataset_idx))
                return cursor.fetchone()[0]


    @staticmethod
    def get_average_accuracy_per_JSON_array_index_group_by_dataset() -> list[tuple[int, int, float]]:
        """This method unrest the JSONB array and assign an index to each element using WITH ORGINALITY
        Followed by grouping by dataset_idx and the array_index
        Calculating the average per index per dataset
        Used for expected performance for each dataset
        """

        query = """
        SELECT 
            dataset_idx,
            ROUND(AVG((elem.value->>'accuracy')::NUMERIC), 4) AS avg_accuracy
        FROM
            hp_evaluations,
            LATERAL jsonb_array_elements(results) WITH ORDINALITY AS elem(value, elem_index)
        GROUP BY
            dataset_idx, elem_index
        ORDER BY 
            dataset_idx ASC, elem_index ASC;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                return rows

    @staticmethod
    def get_average_accuracy_per_JSON_array_index_group_by_model() -> list[tuple[int, int, float]]:
        """This method unrest the JSONB array and assign an index to each element using WITH ORGINALITY
        Followed by grouping by model_idx and the array_index
        Calculating the average per index per model
        Used for expected performance for each model
        """

        query = """
        SELECT 
            model_idx,
            ROUND(AVG((elem.value->>'accuracy')::NUMERIC), 4) AS avg_accuracy
        FROM
            hp_evaluations,
            LATERAL jsonb_array_elements(results) WITH ORDINALITY AS elem(value, elem_index)
        GROUP BY
            model_idx, elem_index
        ORDER BY 
            model_idx ASC, elem_index ASC;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                return rows