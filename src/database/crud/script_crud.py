from src.database.connection import get_connection
from src.database.object import Script
import json

class ScriptRepository:


    @staticmethod
    def create_script(dataset_idx: int, model_idx: int, script_code: str, sgd_config: dict) -> Script | None:
        query = """
        INSERT INTO scripts (dataset_idx, model_idx, script_code, sgd_best_performing_configuration)
        VALUES (%s, %s , %s, %s)
        RETURNING script_idx, dataset_idx, model_idx, script_code, sgd_best_performing_configuration, created_at;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (dataset_idx, model_idx, script_code, json.dumps(sgd_config)))
                row = cursor.fetchone()
                return Script.from_row(row) if row else None
    

    @staticmethod
    def get_script_by_script_idx(script_idx: int) -> Script | None:
        """Retrieve a script by its primary key."""
        query = """
        SELECT script_idx, dataset_idx, model_idx, script_code, sgd_best_performing_configuration, created_at
        FROM scripts
        WHERE script_idx = %s;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (script_idx,))
                row = cursor.fetchone()
                return Script.from_row(row) if row else None
            
    @staticmethod
    def get_script_id_by_model_idx_dataset_idx(model_idx: int, dataset_idx: int) -> int:
        query = """
        SELECT script_idx
        FROM scripts
        WHERE model_idx = %s AND dataset_idx = %s;
        """

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (model_idx, dataset_idx))
                script_idx = cursor.fetchone()
                return script_idx
            

    @staticmethod
    def get_script_by_model_and_dataset_idx(model_idx: int, dataset_idx: int) -> Script | None:
        """Retrieve a script by its primary key."""
        query = """
        SELECT script_idx, dataset_idx, model_idx, script_code, sgd_best_performing_configuration, created_at
        FROM scripts
        WHERE model_idx = %s AND dataset_dix = %s;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (model_idx, dataset_idx))
                row = cursor.fetchone()
                return Script.from_row(row) if row else None


    @staticmethod
    def update_script(script_idx: int, sgd_config: dict) -> bool:
        """Update the script code and its best configuration."""
        query = """
        UPDATE scripts
        SET sgd_best_performing_configuration = %s
        WHERE script_idx = %s;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (json.dumps(sgd_config), script_idx))
                return cursor.rowcount > 0

    @staticmethod
    def delete_script(script_idx: int) -> bool:
        """Delete a script by its primary key."""
        query = """
        DELETE FROM scripts
        WHERE script_idx = %s;
        """
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (script_idx,))
                return cursor.rowcount > 0