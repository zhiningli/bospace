-- raw_sql_migration_scripts/create_datasets_table.sql
CREATE TABLE IF NOT EXISTS datasets (
    dataset_idx SERIAL PRIMARY KEY,
    code TEXT NOT NULL,
    input_size INTEGER NOT NULL,
    num_classes INTEGER NOT NULL,
    meta_features REAL[] NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);