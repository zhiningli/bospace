from src.database.connection import get_connection
from src.database.object import Rank

class RankRepository:
    """CRUD operations for the 'ranks' table."""

    @staticmethod
    def create_rank(
        object_type: str,
        object_1_idx: int,
        object_1: list[float],
        object_2_idx: int,
        object_2: list[float],
        rank: float
    ) -> Rank | None:
        """Create a new rank entry with polymorphic foreign keys."""
        query = """
        INSERT INTO ranks (
            object_type, 
            object_1_idx, object_1,
            object_2_idx, object_2,
            rank
        )
        VALUES (%s::rank_object_enum, %s, %s, %s, %s, %s)
        RETURNING 
            ranks_idx, object_type, 
            object_1_idx, object_1,
            object_2_idx, object_2,
            rank, created_at;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    query,
                    (
                        object_type, 
                        object_1_idx, object_1,
                        object_2_idx, object_2,
                        rank
                    )
                )
                row = cursor.fetchone()
                return Rank.from_row(row) if row else None

    @staticmethod
    def get_results_by_object_type(object_type: str) -> list[Rank]:
        """Retrieve all ranks for a specific object type."""
        query = """
        SELECT 
            *
        FROM ranks
        WHERE object_type = %s::rank_object_enum;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (object_type,))
                rows = cursor.fetchall()
                return [Rank.from_row(row) for row in rows]

    @staticmethod
    def delete_rank(rank_idx: int) -> bool:
        """Delete a rank entry by ID."""
        query = "DELETE FROM ranks WHERE ranks_idx = %s;"
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (rank_idx,))
                return cursor.rowcount > 0
