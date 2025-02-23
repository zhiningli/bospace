-- A summary table based on information from rank tables that show how a dataset is expected to perform against a list of hyperparameter configurations
CREATE MATERIALIZED VIEW dataset_evaluation_summary AS
SELECT 
    dataset_idx,
    elem_index,
    ROUND(AVG((elem.value->>'accuracy')::NUMERIC), 4) AS avg_accuracy
FROM
    hp_evaluations,
    LATERAL jsonb_array_elements(results) WITH ORDINALITY AS elem(value, elem_index)
GROUP BY
    dataset_idx, elem_index
ORDER BY 
    dataset_idx ASC, elem_index ASC
WITH DATA;
