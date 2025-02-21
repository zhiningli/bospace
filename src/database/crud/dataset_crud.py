from src.database.connection import get_connection
from src.database.object import Dataset
import logging

logger = logging.getLogger("database")

class DatasetRepository:

    """CRUD operations for the 'datasets' table using the dataset dataclass """

    @staticmethod
    def create_dataset(code: str, input_size: int, num_classes: int,  meta_features: list[float] = None) -> Dataset:
        query = """
        INSERT INTO datasets (code, input_size, num_classes, meta_features)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (dataset_idx) DO NOTHING
        RETURNING dataset_idx, code, input_size, num_classes, meta_features, created_at;
        """

        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Executing query: {query} with params: -{code}, {input_size}, {num_classes}, {meta_features}")
                    cursor.execute(query, (code, input_size, num_classes, meta_features))
                    row = cursor.fetchone()
                    if row:
                        logger.info(f"Dataset created successfully: dataset_idx={row[0]}")
                        return Dataset.from_row(row)
                    else:
                        logger.warning(f"Dataset already exists or not created")
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}", exc_info=True)
                

    @staticmethod
    def get_dataset(dataset_idx: int) -> Dataset:
        query = """
        SELECT * FROM datasets
        WHERE dataset_idx = %s;
        """

        try: 
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f" Fetching dataset with dataset_idx={dataset_idx}")
                    cursor.execute(query, (dataset_idx,))
                    row = cursor.fetchone()
                    if row:
                        logger.info(f" Dataset retrieved with dataset_idx={dataset_idx}")
                        return Dataset.from_row(row)
                    else:
                        logger.warning(f" No dataset object is found with idx: {dataset_idx}")
                           
        except Exception as e:
            logger.error(f"Failed to retrive dataset {e}", exc_info=True)
                
    @staticmethod
    def get_all_dataset() -> list[Dataset] | None:
        query = """
        SELECT * FROM datasets
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug("Fetching all datasets from table")
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    if rows:
                        logger.info("Datasets fetched from table")
                        return [Dataset.from_row(row) for row in rows if rows]
                    else:
                        logger.warning("No dataset found in the table")
        except Exception as e:
            logger.error(f"Error when retrieving all datasets {e}", exc_info=True)


    @staticmethod
    def update_meta_features(dataset_idx: int, meta_features: list[float]) -> bool:
        query = """
        UPDATE dataset
        SET meta_features = %s
        WHERE dataset_idx = %s
        """     

        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Updating meta_features for dataset_idx={dataset_idx} with {meta_features}")
                    cursor.execute(query, (meta_features, dataset_idx))
                    if cursor.rowcount > 0:
                        logger.info(f"✅ Meta features updated for dataset_idx={dataset_idx}")
                        return True
                    else:
                        logger.warning(f"No dataset found to update: dataset_idx={dataset_idx}")    
        except Exception as e:
            logger.error(f"Failed to update meta features for dataset_idx={dataset_idx}: {e}", exc_info=True)
        
        return False

    @staticmethod
    def delete_dataset(dataset_idx: int) -> bool:
        
        query = """
        DELETE FROM modles
        WHERE dataset_idx = %s;
        """

        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    logger.debug(f"Deleting dataset with dataset_idx={dataset_idx}")
                    cursor.execute(query, (dataset_idx,))

                    if cursor.rowcount > 0:
                        logger.info(f"Dataset deleted successfully: dataset_idx={dataset_idx}")
                        return True
                    else:
                        logger.warning(f"No dataset found to delete: dataset_idx={dataset_idx}")
        except Exception as e:
            logger.error(f"Failed to delete dataset: {e}", exc_info=True)
        return False

