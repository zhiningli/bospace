import logging
from src.database.connection import get_connection
from src.database.object import Model

# Logger for the database
logger = logging.getLogger("database")

class ModelRepository:
    """CRUD operations for the 'models' table using the Model dataclass"""

    @staticmethod
    def create_model(model_idx: int, code: str, feature_vector: list[float] = None) -> Model | None:
        """Create a new model record in the database."""
        query = """
        INSERT INTO models (model_idx, code, feature_vector)
        VALUES (%s, %s, %s)
        ON CONFLICT (model_idx) DO NOTHING
        RETURNING model_idx, code, feature_vector, created_at;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Creating model: model_idx={model_idx}, code={code}, feature_vector={feature_vector}")
                    cursor.execute(query, (model_idx, code, feature_vector))
                    row = cursor.fetchone()

                    if row:
                        logger.info(f"Model created successfully: model_idx={model_idx}")
                        return Model.from_row(row)
                    else:
                        logger.warning(f"Model already exists or could not be created: model_idx={model_idx}")
        except Exception as e:
            logger.error(f"Failed to create model: {e}", exc_info=True)
        return None

    @staticmethod
    def get_model(model_idx: int) -> Model | None:
        """Retrieve a model by its ID."""
        query = """
        SELECT *
        FROM models
        WHERE model_idx = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"🔍 Fetching model with model_idx={model_idx}")
                    cursor.execute(query, (model_idx,))
                    row = cursor.fetchone()

                    if row:
                        logger.info(f"Model retrieved successfully: model_idx={model_idx}")
                        return Model.from_row(row)
                    else:
                        logger.warning(f"No model found with model_idx={model_idx}")
        except Exception as e:
            logger.error(f"Failed to retrieve model: {e}", exc_info=True)
        return None

    @staticmethod
    def get_all_models() -> list[Model] | None:
        """Retrieve all models from the database."""
        query = """
        SELECT *
        FROM models;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug("🔍 Fetching all models.")
                    cursor.execute(query)
                    rows = cursor.fetchall()

                    if rows:
                        logger.info(f"Retrieved {len(rows)} models successfully.")
                        return [Model.from_row(row) for row in rows]
                    else:
                        logger.warning("No models found in the database.")
        except Exception as e:
            logger.error(f"Failed to retrieve all models: {e}", exc_info=True)
        return []

    @staticmethod
    def update_feature_vector(model_idx: int, feature_vector: list[float]) -> bool:
        """Update the feature vector for a specific model."""
        query = """
        UPDATE models
        SET feature_vector = %s
        WHERE model_idx = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Updating feature vector for model_idx={model_idx}")
                    cursor.execute(query, (feature_vector, model_idx))

                    if cursor.rowcount > 0:
                        logger.info(f"Feature vector updated for model_idx={model_idx}")
                        return True
                    else:
                        logger.warning(f"No model found to update: model_idx={model_idx}")
        except Exception as e:
            logger.error(f"Failed to update feature vector for model_idx={model_idx}: {e}", exc_info=True)
        return False

    @staticmethod
    def delete_model(model_idx: int) -> bool:
        """Delete a model by its ID."""
        query = """
        DELETE FROM models
        WHERE model_idx = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Deleting model with model_idx={model_idx}")
                    cursor.execute(query, (model_idx,))

                    if cursor.rowcount > 0:
                        logger.info(f"Model deleted successfully: model_idx={model_idx}")
                        return True
                    else:
                        logger.warning(f"No model found to delete: model_idx={model_idx}")
        except Exception as e:
            logger.error(f"Failed to delete model: {e}", exc_info=True)
        return False
