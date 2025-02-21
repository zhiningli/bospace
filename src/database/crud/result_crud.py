import logging
from src.database.connection import get_connection
from src.database.object import Result
import json

# Logger setup for database operations
logger = logging.getLogger("database")

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
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"🔍 Creating result: script_idx={script_idx}, model_idx={model_idx}, dataset_idx={dataset_idx}, result_type={result_type}")
                    cursor.execute(query, (script_idx, model_idx, dataset_idx, result_type, json.dumps(sgd_config)))
                    row = cursor.fetchone()

                    if row:
                        logger.info(f"Successfully created result for script_idx={script_idx} and result_type={result_type}.")
                        return Result.from_row(row)
                    else:
                        logger.warning(f"Failed to create result for script_idx={script_idx} and result_type={result_type}.")
        except Exception as e:
            logger.error(f"Failed to create result: {e}", exc_info=True)
        return None

    @staticmethod
    def get_result(result_id: int) -> Result | None:
        """Retrieve a specific result by its ID."""
        query = """
        SELECT * FROM results
        WHERE result_id = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"🔍 Fetching result with result_id={result_id}")
                    cursor.execute(query, (result_id,))
                    row = cursor.fetchone()

                    if row:
                        logger.info(f"Retrieved result with result_id={result_id}.")
                        return Result.from_row(row)
                    else:
                        logger.warning(f"No result found with result_id={result_id}.")
        except Exception as e:
            logger.error(f"Failed to retrieve result with result_id={result_id}: {e}", exc_info=True)
        return None

    @staticmethod
    def get_results_by_script(script_idx: int) -> list[Result]:
        """Retrieve all results for a specific script."""
        query = """
        SELECT * FROM results
        WHERE script_idx = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"🔍 Fetching results for script_idx={script_idx}")
                    cursor.execute(query, (script_idx,))
                    rows = cursor.fetchall()

                    if rows:
                        logger.info(f"Retrieved {len(rows)} results for script_idx={script_idx}.")
                        return [Result.from_row(row) for row in rows]
                    else:
                        logger.warning(f"No results found for script_idx={script_idx}.")
        except Exception as e:
            logger.error(f"Failed to retrieve results for script_idx={script_idx}: {e}", exc_info=True)
        return []

    @staticmethod
    def get_results_by_script_and_result_type(script_idx: int, result_type: str) -> Result | None:
        """Retrieve results for a specific script and result type."""
        query = """
        SELECT * FROM results
        WHERE script_idx = %s AND result_type = %s::result_type_enum;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Fetching results for script_idx={script_idx} and result_type={result_type}")
                    cursor.execute(query, (script_idx, result_type))
                    row = cursor.fetchone()

                    if row:
                        logger.info(f"Retrieved result for script_idx={script_idx} and result_type={result_type}.")
                        return Result.from_row(row)
                    else:
                        logger.warning(f"No result found for script_idx={script_idx} and result_type={result_type}.")
        except Exception as e:
            logger.error(f"Failed to retrieve results for script_idx={script_idx} and result_type={result_type}: {e}", exc_info=True)
        return None

    @staticmethod
    def update_sgd_config(script_idx: int, result_type: str, new_config: dict) -> bool:
        """Update the SGD best-performing configuration for a specific script and result type."""
        query = """
        UPDATE results
        SET sgd_best_performing_configuration = %s
        WHERE script_idx = %s AND result_type = %s::result_type_enum;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Updating SGD config for script_idx={script_idx} and result_type={result_type}")
                    cursor.execute(query, (json.dumps(new_config), script_idx, result_type))

                    if cursor.rowcount > 0:
                        logger.info(f"Successfully updated SGD config for script_idx={script_idx} and result_type={result_type}.")
                        return True
                    else:
                        logger.warning(f"No result found to update for script_idx={script_idx} and result_type={result_type}.")
        except Exception as e:
            logger.error(f"Failed to update SGD config: {e}", exc_info=True)
        return False

    @staticmethod
    def delete_result(result_id: int) -> bool:
        """Delete a result by its ID."""
        query = """
        DELETE FROM results
        WHERE result_id = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Deleting result with result_id={result_id}")
                    cursor.execute(query, (result_id,))

                    if cursor.rowcount > 0:
                        logger.info(f"Successfully deleted result with result_id={result_id}.")
                        return True
                    else:
                        logger.warning(f"No result found to delete with result_id={result_id}.")
        except Exception as e:
            logger.error(f"Failed to delete result with result_id={result_id}: {e}", exc_info=True)
        return False
