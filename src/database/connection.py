import psycopg2
import os
import logging
from dotenv import load_dotenv


load_dotenv()  # Ensure this is called to load .env variables


logger = logging.getLogger("database")
logger.setLevel(logging.DEBUG)

def get_connection():
    """Establish a database connection based on the environment."""
    env = os.getenv("ENV", "dev").lower()
    if env not in ["dev", "test"]:
        raise ValueError(f"Invalid ENV value: {env}. Expected 'dev' or 'test'.")

    # Database configuration based on environment
    db_name = os.getenv(f"{env.upper()}_DB_NAME")
    user = os.getenv(f"{env.upper()}_DB_USER")
    password = os.getenv(f"{env.upper()}_DB_PASSWORD")
    host = os.getenv(f"{env.upper()}_DB_HOST")
    port = os.getenv(f"{env.upper()}_DB_PORT")

    if not all([db_name, user, password, host, port]):
        raise ValueError(f"Missing database configuration for environment '{env}'.")
    logger.debug(f"Connecting to {env} database: {db_name} at {host}:{port} as user '{user}'")

    try:
        connection = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port,
            connect_timeout=10
        )
        logger.info(f"Successfully connected to {env} database: {db_name}.")
        return connection

    except psycopg2.OperationalError as e:
        logger.error(f"Failed to connect to {env} database: {e}")
        raise


