-- raw_sql_migration_scripts/create_hp_evaluations_table.sql
CREATE TABLE IF NOT EXISTS hp_evaluations (
    hp_evaluation_id SERIAL PRIMARY KEY,
    model_idx INTEGER NOT NULL,
    dataset_idx INTEGER NOT NULL,
    results JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_idx) REFERENCES models(model_idx) ON DELETE CASCADE,
    FOREIGN KEY (dataset_idx) REFERENCES datasets(dataset_idx) ON DELETE CASCADE
);