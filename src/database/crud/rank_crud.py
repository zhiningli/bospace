from src.database.connection import get_connection
from src.database.object import Similarity

class SimilarityRepository:
    """CRUD operations for the 'ranks' table."""

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

        with get_connection() as conn:
            with conn.cursor() as cursor:
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
                return Similarity.from_row(row) if row else None

    @staticmethod
    def get_results_by_object_type(object_type: str) -> list[Similarity]:
        """Retrieve all similarities for a specific object type."""
        query = """
        SELECT 
            *
        FROM similarities
        WHERE object_type = %s::similarity_object_enum;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (object_type,))
                rows = cursor.fetchall()
                return [Similarity.from_row(row) for row in rows]

    @staticmethod
    def delete_rank(similarity_idx: int) -> bool:
        """Delete a similarity entry by ID."""
        query = "DELETE FROM similarities WHERE similarity_idx = %s;"
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (similarity_idx,))
                return cursor.rowcount > 0
            
    @staticmethod
    def exists_similarity(object_type: str, object_1_idx: int, object_2_idx: int) -> bool:
        """Check if a model_idx and dataset_idx pair exists in the repository."""
        query = """
        SELECT EXISTS(
            SELECT 1 
            FROM similarities
            WHERE object_type = %s::rank_object_enum AND object_1_idx = %s AND object_2_idx = %s  
            AND dataset_idx = %s
        );
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (object_type, object_1_idx, object_2_idx))
                return cursor.fetchone()[0]
