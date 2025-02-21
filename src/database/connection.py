import psycopg2
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger("database")

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

    logger.debug(f"Attempting to connect to database at {DB_PARAMS['host']:{DB_PARAMS['port']}}")
    
    try:
        connection = psycopg2.connect(**DB_PARAMS)

        logger.info(f"Successfully connected to database '{DB_PARAMS['dbname']}' as user '{DB_PARAMS['user']}'.")

        return connection
    except psycopg2.OperationalError as op_err:
        logger.error(f"OperationalError: Unable to connect to the database - {op_err}")
    except Exception as e:
        logger.critical(f"Unexpected error while connecting to the database: {e}")
