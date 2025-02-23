-- A summary table based on information from rank tables that show how a model is expected to perform against a list of hyperparameter configurations
CREATE MATERIALIZED VIEW model_evaluation_summary AS
SELECT 
    model_idx,
    elem_index,
    ROUND(AVG((elem.value->>'accuracy')::NUMERIC), 4) AS avg_accuracy
FROM
    hp_evaluations,
    LATERAL jsonb_array_elements(results) WITH ORDINALITY AS elem(value, elem_index)
GROUP BY
    model_idx, elem_index
ORDER BY 
    model_idx ASC, elem_index ASC
WITH DATA;
