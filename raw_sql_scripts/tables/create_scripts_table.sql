-- raw_sql_migration_scripts/create_scripts_table.sql
CREATE TABLE IF NOT EXISTS scripts (
    script_idx SERIAL PRIMARY KEY,
    dataset_idx INT NOT NULL,
    model_idx INT NOT NULL,
    script_code TEXT NOT NULL,
    sgd_best_performing_configuration JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_dataset
        FOREIGN KEY (dataset_idx)
        REFERENCES datasets (dataset_idx)
        ON DELETE CASCADE,
    CONSTRAINT fk_model
        FOREIGN KEY (model_idx)
        REFERENCES models (model_idx)
        ON DELETE CASCADE
);