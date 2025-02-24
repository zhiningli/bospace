import json
import logging
from src.database.connection import get_connection
from src.database.object import HP_evaluation

# Logger setup for database operations
logger = logging.getLogger("database")

class HPEvaluationRepository:
    """CRUD operations for the hp_evaluations table."""

    @staticmethod
    def create_hp_evaluation(model_idx: int, dataset_idx: int, results: list[dict]) -> HP_evaluation | None:
        """Create a new hyperparameter evaluation record."""
        query = """
        INSERT INTO hp_evaluations (model_idx, dataset_idx, results)
        VALUES (%s, %s, %s)
        ON CONFLICT (dataset_idx, model_idx)
        DO UPDATE SET results = EXCLUDED.results
        RETURNING hp_evaluation_id, model_idx, dataset_idx, results, created_at;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"🔍 Executing query: {query} with params: model_idx={model_idx}, dataset_idx={dataset_idx}, results={results}")
                    cursor.execute(query, (model_idx, dataset_idx, json.dumps(results)))
                    row = cursor.fetchone()

                    if row:
                        logger.info(f"HP evaluation created successfully for model_idx={model_idx}, dataset_idx={dataset_idx}.")
                        return HP_evaluation.from_row(row)
                    else:
                        logger.warning(f"No HP evaluation created for model_idx={model_idx}, dataset_idx={dataset_idx}.")
                        return 
        except Exception as e:
            logger.error(f"Failed to create HP evaluation: {e}", exc_info=True)
        return None

    @staticmethod
    def get_hp_evaluation_by_id(hp_evaluation_id: int) -> HP_evaluation | None:
        """Retrieve an HP evaluation by its ID."""
        query = """
        SELECT * FROM hp_evaluations
        WHERE hp_evaluation_id = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Fetching HP evaluation with hp_evaluation_id={hp_evaluation_id}")
                    cursor.execute(query, (hp_evaluation_id,))
                    row = cursor.fetchone()

                    if row:
                        logger.info(f"HP evaluation retrieved: id={hp_evaluation_id}.")
                        return HP_evaluation.from_row(row)
                    else:
                        logger.warning(f"No HP evaluation found with id={hp_evaluation_id}.")
        except Exception as e:
            logger.error(f"Failed to retrieve HP evaluation: {e}", exc_info=True)
        return None

    @staticmethod
    def update_results(hp_evaluation_id: int, new_results: list[dict]) -> bool:
        """Update results for an existing HP evaluation."""
        query = """
        UPDATE hp_evaluations
        SET results = %s
        WHERE hp_evaluation_id = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Updating results for hp_evaluation_id={hp_evaluation_id} with new_results={new_results}")
                    cursor.execute(query, (json.dumps(new_results), hp_evaluation_id))

                    if cursor.rowcount > 0:
                        logger.info(f"Successfully updated HP evaluation: id={hp_evaluation_id}.")
                        return True
                    else:
                        logger.warning(f"No HP evaluation found to update with id={hp_evaluation_id}.")
        except Exception as e:
            logger.error(f"Failed to update HP evaluation: {e}", exc_info=True)
        return False

    @staticmethod
    def delete_hp_evaluation(hp_evaluation_id: int) -> bool:
        """Delete an HP evaluation by its ID."""
        query = """
        DELETE FROM hp_evaluations
        WHERE hp_evaluation_id = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Deleting HP evaluation with hp_evaluation_id={hp_evaluation_id}")
                    cursor.execute(query, (hp_evaluation_id,))

                    if cursor.rowcount > 0:
                        logger.info(f"Successfully deleted HP evaluation: id={hp_evaluation_id}.")
                        return True
                    else:
                        logger.warning(f"No HP evaluation found to delete with id={hp_evaluation_id}.")
        except Exception as e:
            logger.error(f"Failed to delete HP evaluation: {e}", exc_info=True)
        return False

    @staticmethod
    def exists_hp_evaluation(model_idx: int, dataset_idx: int) -> bool:
        """Check if an evaluation exists for a given model and dataset."""
        query = """
        SELECT EXISTS(
            SELECT 1 
            FROM hp_evaluations
            WHERE model_idx = %s AND dataset_idx = %s
        );
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"🔍 Checking existence of HP evaluation for model_idx={model_idx}, dataset_idx={dataset_idx}")
                    cursor.execute(query, (model_idx, dataset_idx))
                    exists = cursor.fetchone()[0]

                    if exists:
                        logger.info(f"HP evaluation exists for model_idx={model_idx}, dataset_idx={dataset_idx}.")
                    else:
                        logger.warning(f"No HP evaluation exists for model_idx={model_idx}, dataset_idx={dataset_idx}.")
                    return exists
        except Exception as e:
            logger.error(f"Failed to check existence of HP evaluation: {e}", exc_info=True)
        return False

    @staticmethod
    def get_average_accuracy_per_JSON_array_index_group_by_dataset() -> list[tuple[int, int, float]]:
        """Calculate the average accuracy per JSON array index, grouped by dataset."""
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
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Calculating average accuracy per index grouped by dataset.")
                    cursor.execute(query)
                    rows = cursor.fetchall()

                    if rows:
                        logger.info(f"Retrieved {len(rows)} dataset-level average accuracy records.")
                    else:
                        logger.warning("No results found for dataset-level accuracy averages.")
                    return rows
        except Exception as e:
            logger.error(f"Failed to calculate dataset-level average accuracy: {e}", exc_info=True)
        return []

    @staticmethod
    def get_average_accuracy_per_JSON_array_index_group_by_model() -> list[tuple[int, int, float]]:
        """Calculate the average accuracy per JSON array index, grouped by model."""
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
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Calculating average accuracy per index grouped by model.")
                    cursor.execute(query)
                    rows = cursor.fetchall()

                    if rows:
                        logger.info(f"Retrieved {len(rows)} model-level average accuracy records.")
                    else:
                        logger.warning("No results found for model-level accuracy averages.")
                    return rows
        except Exception as e:
            logger.error(f"Failed to calculate model-level average accuracy: {e}", exc_info=True)
        return []
