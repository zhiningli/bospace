-- Step 1: Identify duplicate entries
SELECT dataset_idx, model_idx, COUNT(*)
FROM hp_evaluations
GROUP BY dataset_idx, model_idx
HAVING COUNT(*) > 1;

-- Step 2: Remove duplicate entries, keeping only one
DELETE FROM hp_evaluations
WHERE id NOT IN (
    SELECT MIN(id)
    FROM hp_evaluations
    GROUP BY dataset_idx, model_idx
);

-- Step 3: Add unique constraint
ALTER TABLE hp_evaluations
ADD CONSTRAINT unique_dataset_model
UNIQUE (dataset_idx, model_idx);
