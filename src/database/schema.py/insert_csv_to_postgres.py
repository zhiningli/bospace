
import csv
import json
from src.database.connection import get_connection

def insert_csv_to_postgres():

    with get_connection() as conn:
        with conn.cursor() as cursor:
            with open("dataset.csv", "r") as csv_file:
                reader = csv.DictReader(csv_file)

                for row in reader:
                    dataset_idx = int(row["dataset_idx"])
                    code = json.loads(row["code"])
                    meta_features = [float(x) for x in row["meta_features"].split(",")]

                
                    insert_query = """
                    INSERT INTO datasets (dataset_idx, code, meta_features)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (dataset_idx) DO NOTHING;
                    """

                    cursor.execute(insert_query, (dataset_idx, json.dumps(code), meta_features))

                conn.commit()
                print("CSV data imported into PostgreSQL successfully")


if __name__ == "__main__":
    insert_csv_to_postgres()