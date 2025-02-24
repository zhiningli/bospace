-- raw_sql_migration_scripts/create_similarities_table.sql
CREATE TABLE dataset_similarities (
    similarity_id SERIAL PRIMARY KEY,
    object_1_idx INT NOT NULL,
    object_1_feature REAL[] NOT NULL,
    object_2_idx INT NOT NULL,
    object_2_feature REAL[] NOT NULL,
    similarity FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign keys to datasets
    CONSTRAINT fk_dataset_1 FOREIGN KEY (object_1_idx) REFERENCES datasets(dataset_idx) ON DELETE CASCADE,
    CONSTRAINT fk_dataset_2 FOREIGN KEY (object_2_idx) REFERENCES datasets(dataset_idx) ON DELETE CASCADE
);

CREATE TABLE model_similarities (
    similarity_id SERIAL PRIMARY KEY,
    object_1_idx INT NOT NULL,
    object_1_feature REAL[] NOT NULL,
    object_2_idx INT NOT NULL,
    object_2_feature REAL[] NOT NULL,
    similarity FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign keys to models
    CONSTRAINT fk_model_1 FOREIGN KEY (object_1_idx) REFERENCES models(model_idx) ON DELETE CASCADE,
    CONSTRAINT fk_model_2 FOREIGN KEY (object_2_idx) REFERENCES models(model_idx) ON DELETE CASCADE
);
