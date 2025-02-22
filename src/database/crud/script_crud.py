import logging
from src.database.connection import get_connection
from src.database.object import Script
import json

# Logger for database operations
logger = logging.getLogger("database")

class ScriptRepository:

    @staticmethod
    def create_script(dataset_idx: int, model_idx: int, script_code: str, sgd_config: dict) -> Script | None:
        """Create a new script entry."""
        query = """
        INSERT INTO scripts (dataset_idx, model_idx, script_code, sgd_best_performing_configuration)
        VALUES (%s, %s, %s, %s)
        RETURNING script_idx, dataset_idx, model_idx, script_code, sgd_best_performing_configuration, created_at;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Creating script for dataset_idx={dataset_idx} and model_idx={model_idx}")
                    cursor.execute(query, (dataset_idx, model_idx, script_code, json.dumps(sgd_config)))
                    row = cursor.fetchone()

                    if row:
                        logger.info(f"Successfully created script for dataset_idx={dataset_idx} and model_idx={model_idx}.")
                        return Script.from_row(row)
                    else:
                        logger.warning(f"Failed to create script for dataset_idx={dataset_idx} and model_idx={model_idx}.")
        except Exception as e:
            logger.error(f"Failed to create script: {e}", exc_info=True)
        return None

    @staticmethod
    def get_script_by_script_idx(script_idx: int) -> Script | None:
        """Retrieve a script by its primary key."""
        query = """
        SELECT script_idx, dataset_idx, model_idx, script_code, sgd_best_performing_configuration, created_at
        FROM scripts
        WHERE script_idx = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Fetching script with script_idx={script_idx}")
                    cursor.execute(query, (script_idx,))
                    row = cursor.fetchone()

                    if row:
                        logger.info(f"Retrieved script with script_idx={script_idx}.")
                        return Script.from_row(row)
                    else:
                        logger.warning(f"No script found with script_idx={script_idx}.")
        except Exception as e:
            logger.error(f"Failed to retrieve script with script_idx={script_idx}: {e}", exc_info=True)
        return None

    @staticmethod
    def get_script_id_by_model_idx_dataset_idx(model_idx: int, dataset_idx: int) -> int | None:
        """Get script ID by model and dataset index."""
        query = """
        SELECT script_idx
        FROM scripts
        WHERE model_idx = %s AND dataset_idx = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"🔍 Fetching script ID for model_idx={model_idx} and dataset_idx={dataset_idx}")
                    cursor.execute(query, (model_idx, dataset_idx))
                    script_idx = cursor.fetchone()

                    if script_idx:
                        logger.info(f"Found script_idx={script_idx[0]} for model_idx={model_idx} and dataset_idx={dataset_idx}.")
                        return script_idx[0]
                    else:
                        logger.warning(f"No script found for model_idx={model_idx} and dataset_idx={dataset_idx}.")
        except Exception as e:
            logger.error(f"Failed to get script ID for model_idx={model_idx} and dataset_idx={dataset_idx}: {e}", exc_info=True)
        return None

    @staticmethod
    def get_script_by_model_and_dataset_idx(model_idx: int, dataset_idx: int) -> Script | None:
        """Retrieve a script by model and dataset indices."""
        query = """
        SELECT script_idx, dataset_idx, model_idx, script_code, sgd_best_performing_configuration, created_at
        FROM scripts
        WHERE model_idx = %s AND dataset_idx = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Fetching script for model_idx={model_idx} and dataset_idx={dataset_idx}")
                    cursor.execute(query, (model_idx, dataset_idx))
                    row = cursor.fetchone()

                    if row:
                        logger.info(f"✅ Retrieved script for model_idx={model_idx} and dataset_idx={dataset_idx}.")
                        return Script.from_row(row)
                    else:
                        logger.warning(f"No script found for model_idx={model_idx} and dataset_idx={dataset_idx}.")
        except Exception as e:
            logger.error(f"Failed to retrieve script for model_idx={model_idx} and dataset_idx={dataset_idx}: {e}", exc_info=True)
        return None

    @staticmethod
    def update_script_code(script_idx: int, script_code: str) -> bool:
        """Update the script code for a given script ID."""
        query = """
        UPDATE scripts
        SET script_code = %s
        WHERE script_idx = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"🛠️ Updating script code for script_idx={script_idx}")
                    cursor.execute(query, (script_code, script_idx))

                    if cursor.rowcount > 0:
                        logger.info(f"Successfully updated script code for script_idx={script_idx}.")
                        return True
                    else:
                        logger.warning(f"No script found to update for script_idx={script_idx}.")
        except Exception as e:
            logger.error(f"Failed to update script code for script_idx={script_idx}: {e}", exc_info=True)
        return False

    @staticmethod
    def update_script_sgd_config(script_idx: int, sgd_config: dict) -> bool:
        """Update the SGD configuration for a script."""
        query = """
        UPDATE scripts
        SET sgd_best_performing_configuration = %s
        WHERE script_idx = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"🛠️ Updating SGD config for script_idx={script_idx}")
                    cursor.execute(query, (json.dumps(sgd_config), script_idx))

                    if cursor.rowcount > 0:
                        logger.info(f"Successfully updated SGD config for script_idx={script_idx}.")
                        return True
                    else:
                        logger.warning(f"No script found to update for script_idx={script_idx}.")
        except Exception as e:
            logger.error(f"Failed to update SGD config for script_idx={script_idx}: {e}", exc_info=True)
        return False

    @staticmethod
    def delete_script(script_idx: int) -> bool:
        """Delete a script by its primary key."""
        query = """
        DELETE FROM scripts
        WHERE script_idx = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Deleting script with script_idx={script_idx}")
                    cursor.execute(query, (script_idx,))

                    if cursor.rowcount > 0:
                        logger.info(f"Successfully deleted script with script_idx={script_idx}.")
                        return True
                    else:
                        logger.warning(f"No script found to delete with script_idx={script_idx}.")
        except Exception as e:
            logger.error(f"Failed to delete script with script_idx={script_idx}: {e}", exc_info=True)
        return False
