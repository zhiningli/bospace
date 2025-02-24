import logging
from src.database.connection import get_connection
from src.database.object import Similarity
from datetime import datetime
# Logger setup for database operations
logger = logging.getLogger("database")

logger = logging.getLogger(__name__)

class SimilarityRepository:


    @staticmethod
    def create_similarity(object_type: str,  object_1_idx: int, object_1_feature : list[float], object_2_idx: int, object_2_feature : list[float],  similarity: float):
        """Create similarity entry in the appropriate table based on object_type."""
        if object_type == "dataset":
            table_name = "dataset_similarities"
        elif object_type == "model":
            table_name = "model_similarities"
        else:
            raise ValueError(f"Invalid object_type: {object_type}")

        query = f"""
        INSERT INTO {table_name} (object_1_idx, object_1_feature, object_2_idx, object_2_feature, similarity)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        """

        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (object_1_idx, object_1_feature, object_2_idx, object_2_feature, similarity))
                    conn.commit()
                    logger.info(f"Successfully created {object_type} similarity: {object_1_idx} <-> {object_2_idx}")
        except Exception as e:
            logger.error(f"Failed to create {object_type} similarity: {e}", exc_info=True)


    @staticmethod
    def get_results_by_object_type(object_type: str) -> list[Similarity]:
        """Retrieve all similarities for a specific object type (dataset or model)."""

        # Map object_type to the corresponding table
        table_map = {
            "dataset": "dataset_similarities",
            "model": "model_similarities"
        }

        table_name = table_map.get(object_type.lower())

        if not table_name:
            raise ValueError(f"Invalid object_type: {object_type}. Expected 'dataset' or 'model'.")

        query = f"""
        SELECT 
            *
        FROM {table_name};
        """

        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Fetching all similarities for object_type={object_type} from {table_name}")
                    cursor.execute(query)
                    rows = cursor.fetchall()

                    if rows:
                        logger.info(f"Retrieved {len(rows)} similarity records for object_type={object_type}.")
                        return [Similarity.from_row(row) for row in rows]
                    else:
                        logger.warning(f"No similarities found for object_type={object_type}.")
        except Exception as e:
            logger.error(f"Failed to retrieve similarities for object_type={object_type}: {e}", exc_info=True)

        return []

    @staticmethod
    def get_results_after_time(object_type: str, created_after: str) -> list[Similarity]:
        """Retrieve new similarity records created after a given timestamp for a specific object type.

        Args:
            object_type (str): The type of object ('model' or 'dataset').
            created_after (str): The timestamp (ISO format) to filter records.

        Returns:
            list[Similarity]: A list of similarity objects created after the given timestamp.
        """
        # Map object type to the corresponding table
        table_map = {
            "dataset": "dataset_similarities",
            "model": "model_similarities"
        }

        # Determine the correct table based on object_type
        table_name = table_map.get(object_type.lower())
        if not table_name:
            raise ValueError(f"Invalid object_type: {object_type}. Expected 'dataset' or 'model'.")

        # SQL query using the appropriate table
        query = f"""
        SELECT *
        FROM {table_name}
        WHERE created_at > %s;
        """

        try:
            # Validate timestamp format
            datetime.fromisoformat(created_after)

            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Fetching similarities from {table_name} created after {created_after}.")
                    cursor.execute(query, (created_after,))
                    rows = cursor.fetchall()

                    if rows:
                        logger.info(f"Retrieved {len(rows)} new similarity records for {object_type} after {created_after}.")
                        return [Similarity.from_row(row) for row in rows]
                    else:
                        logger.warning(f"No new similarities found for {object_type} after {created_after}.")
        except ValueError:
            logger.error(f"Invalid timestamp format: {created_after}. Expected ISO format (YYYY-MM-DDTHH:MM:SS).")
        except Exception as e:
            logger.error(f"Failed to retrieve new similarities for {object_type} after {created_after}: {e}", exc_info=True)

        return []




    @staticmethod
    def exists_similarity(object_type: str, object_1_idx: int, object_2_idx: int) -> bool:
        """Check if a similarity entry exists between two objects of the specified type."""

        # Map object type to the corresponding table
        table_map = {
            "dataset": "dataset_similarities",
            "model": "model_similarities"
        }

        # Get the correct table name
        table_name = table_map.get(object_type.lower())
        if not table_name:
            raise ValueError(f"Invalid object_type: {object_type}. Expected 'dataset' or 'model'.")

        # Query to check existence
        query = f"""
        SELECT EXISTS(
            SELECT 1 
            FROM {table_name}
            WHERE object_1_idx = %s 
            AND object_2_idx = %s
        );
        """

        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"🔍 Checking existence of similarity in {table_name} for object_1_idx={object_1_idx}, object_2_idx={object_2_idx}")
                    cursor.execute(query, (object_1_idx, object_2_idx))
                    exists = cursor.fetchone()[0]
                    return exists
        except Exception as e:
            logger.error(f"Failed to check existence of similarity in {table_name}: {e}", exc_info=True)
            return False
