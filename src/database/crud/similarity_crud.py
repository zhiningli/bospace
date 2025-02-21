import logging
from src.database.connection import get_connection
from src.database.object import Similarity

# Logger setup for database operations
logger = logging.getLogger("database")

class SimilarityRepository:
    """CRUD operations for the 'similarities' table."""

    @staticmethod
    def create_similarity(
        object_type: str,
        object_1_idx: int,
        object_1: list[float],
        object_2_idx: int,
        object_2: list[float],
        similarity: float
    ) -> Similarity | None:
        """Create a new similarity entry with polymorphic foreign keys."""
        query = """
        INSERT INTO similarities (
            object_type, 
            object_1_idx, object_1,
            object_2_idx, object_2,
            similarity
        )
        VALUES (%s::similarity_object_enum, %s, %s, %s, %s, %s)
        RETURNING 
            similarity_idx, object_type, 
            object_1_idx, object_1,
            object_2_idx, object_2,
            similarity, created_at;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Creating similarity: object_type={object_type}, object_1_idx={object_1_idx}, object_2_idx={object_2_idx}, similarity={similarity}")
                    cursor.execute(
                        query,
                        (
                            object_type, 
                            object_1_idx, object_1,
                            object_2_idx, object_2,
                            similarity,
                        )
                    )
                    row = cursor.fetchone()

                    if row:
                        logger.info(f"Similarity created successfully for object_1_idx={object_1_idx} and object_2_idx={object_2_idx}.")
                        return Similarity.from_row(row)
                    else:
                        logger.warning(f"Failed to create similarity for object_1_idx={object_1_idx} and object_2_idx={object_2_idx}.")
        except Exception as e:
            logger.error(f"Failed to create similarity: {e}", exc_info=True)
        return None

    @staticmethod
    def get_results_by_object_type(object_type: str) -> list[Similarity]:
        """Retrieve all similarities for a specific object type."""
        query = """
        SELECT 
            *
        FROM similarities
        WHERE object_type = %s::similarity_object_enum;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Fetching all similarities for object_type={object_type}")
                    cursor.execute(query, (object_type,))
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
    def delete_similarity(similarity_idx: int) -> bool:
        """Delete a similarity entry by ID."""
        query = "DELETE FROM similarities WHERE similarity_idx = %s;"
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Deleting similarity with similarity_idx={similarity_idx}")
                    cursor.execute(query, (similarity_idx,))

                    if cursor.rowcount > 0:
                        logger.info(f"Successfully deleted similarity with similarity_idx={similarity_idx}.")
                        return True
                    else:
                        logger.warning(f"No similarity found to delete with similarity_idx={similarity_idx}.")
        except Exception as e:
            logger.error(f"Failed to delete similarity: {e}", exc_info=True)
        return False

    @staticmethod
    def exists_similarity(object_type: str, object_1_idx: int, object_2_idx: int) -> bool:
        """Check if a similarity entry exists between two objects."""
        query = """
        SELECT EXISTS(
            SELECT 1 
            FROM similarities
            WHERE object_type = %s::similarity_object_enum 
            AND object_1_idx = %s 
            AND object_2_idx = %s
        );
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"🔍 Checking existence of similarity for object_type={object_type}, object_1_idx={object_1_idx}, object_2_idx={object_2_idx}")
                    cursor.execute(query, (object_type, object_1_idx, object_2_idx))
                    exists = cursor.fetchone()[0]

                    if exists:
                        logger.info(f"Similarity exists between object_1_idx={object_1_idx} and object_2_idx={object_2_idx}.")
                    else:
                        logger.warning(f"No similarity found between object_1_idx={object_1_idx} and object_2_idx={object_2_idx}.")
                    return exists
        except Exception as e:
            logger.error(f"Failed to check existence of similarity: {e}", exc_info=True)
        return False
