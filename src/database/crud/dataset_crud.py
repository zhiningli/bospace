from src.database.connection import get_connection
from src.database.object import Dataset
class DatasetRepository:

    """CRUD operations for the 'datasets' table using the dataset dataclass """

    @staticmethod
    def create_dataset(dataset_idx: str, code: str, meta_features: list[float] = None) -> Dataset:
        query = """
        INSERT INTO datasets (dataset_idx, code, meta_features)
        VALUES (%s, %s, %s)
        ON CONFLICT (dataset_idx) DO NOTHING
        RETURNING dataset_idx, code, meta_features, created_at;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (dataset_idx, code, meta_features))
                row = cursor.fetchone()
                if row:
                    return Dataset.from_row(row)
                

    @staticmethod
    def get_dataset(dataset_idx: int) -> Dataset:
        query = """
        SELECT * FROM datasets
        WHERE dataset_idx = %s;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (dataset_idx,))
                row = cursor.fetchone()
                if row:
                    return Dataset.from_row(row)   


    @staticmethod
    def update_meta_features(dataset_idx: int, meta_features: list[float]) -> bool:
        query = """
        UPDATE dataset
        SET meta_features = %s
        WHERE dataset_idx = %s
        """     

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (meta_features, dataset_idx))
                if cursor.rowcount > 0:
                    return True
        return False

    @staticmethod
    def delete_dataset(dataset_idx: int) -> bool:
        
        query = """
        DELETE FROM modles
        WHERE dataset_idx = %s;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (dataset_idx, ))
                if cursor.rowcount > 0:
                    return True
                
        return False


