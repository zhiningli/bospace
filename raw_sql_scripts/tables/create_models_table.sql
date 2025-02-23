-- raw_sql_migration_scripts/create_models_table.sql
CREATE TABLE IF NOT EXISTS models (
    model_idx SERIAL PRIMARY KEY,
    code TEXT NOT NULL,
    feature_vector REAL[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);