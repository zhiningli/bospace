import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

DB_name = os.getenv("DB_NAME")
DB_user = os.getenv("DB_USER")
DB_password = os.getenv("DB_PASSWORD")
DB_host = os.getenv("DB_HOST")
DB_port = os.getenv("DB_PORT")

DB_PARAMS = {
    "dbname": DB_name,
    "user": DB_user,
    "password": DB_password,
    "host": DB_host,
    "port": DB_port
}

def get_connection():
    try:
        connection = psycopg2.connect(**DB_PARAMS)
        return connection
    except Exception as e:
        return None
