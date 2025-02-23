# src/database/evaluation_view_crud.py
import logging
from src.database.connection import get_connection
from src.database.object import DatasetEvaluationSummary, ModelEvaluationSummary

logger = logging.getLogger("evaluation_crud")


class EvaluationMaterialisedView:

    @staticmethod
    def get_evaluations_for_all_dataset() -> list[DatasetEvaluationSummary]:
        """Retrieve all dataset evaluation summaries."""
        query = "SELECT * FROM dataset_evaluation_summary;"
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    return [DatasetEvaluationSummary.from_row(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve evaluations: {e}", exc_info=True)
            return []
        
    @staticmethod
    def get_evaluations_for_all_models() -> list[ModelEvaluationSummary]:
        """Retrieve all model evaluation summaries."""
        query = "SELECT * FROM model_evaluation_summary;"
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    return [ModelEvaluationSummary.from_row(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve evaluations: {e}", exc_info=True)
            return []

    @staticmethod
    def refresh_materialized_view():
        """Refresh the materialized views to reflect the latest data."""
        queries = [
            "REFRESH MATERIALIZED VIEW CONCURRENTLY dataset_evaluation_summary;",
            "REFRESH MATERIALIZED VIEW CONCURRENTLY model_evaluation_summary;"
        ]

        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    for query in queries:
                        cursor.execute(query)
                    conn.commit()
                    logger.info("Materialized views refreshed successfully.")
        except Exception as e:
            logger.error(f"Failed to refresh materialized views: {e}", exc_info=True)
