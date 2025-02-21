from src.database.connection import get_connection
from src.database.object import Model

class ModelRepository:

    """CRUD operations for the 'models' table using the Model dataclass"""


    @staticmethod
    def create_model(model_idx: int, code: str, feature_vector: list[float] = None) -> Model:
        query = """
        INSERT INTO models (model_idx, code, feature_vector)
        VALUES (%s, %s, %s)
        ON CONFLICT (model_idx) DO NOTHING
        RETURNING model_idx, code, feature_vector, created_at;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (model_idx, code, feature_vector))
                row = cursor.fetchone()
                if row:
                    return Model.from_row(row)

    @staticmethod
    def get_model(model_idx: int) -> Model:
        query = """
        SELECT *
        FROM models
        WHERE model_idx = %s;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (model_idx,))
                row = cursor.fetchone()
                if row:
                    return Model.from_row(row)
                
    @staticmethod
    def get_all_models_with_ids() -> list[tuple[int, str]] | None:
        query = """
        SELECT model_idx, code
        FROM models;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                return [(row[0], row[1]) for row in rows] if rows else []
                
    @staticmethod
    def update_feature_vector(model_idx: int, feature_vector: list[float]) -> bool:

        query = """
        UPDATE models
        SET feature_vector = %s
        WHERE model_idx = %s;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (feature_vector, model_idx))
                if cursor.rowcount > 0:
                    return True
        return False
    
    @staticmethod
    def delete_model(model_idx: int) -> bool:
        
        query = """
        DELETE FROM modles
        WHERE model_idx = %s;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (model_idx, ))
                if cursor.rowcount > 0:
                    return True
                
        return False