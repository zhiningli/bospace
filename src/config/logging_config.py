from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import os
import logging
import logging.config

# Log format
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s:%(lineno)d] - %(message)s"

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG"
        },
        "file_database": {
            "class": "logging.FileHandler",
            "filename": "logs/database.log",
            "formatter": "default",
            "level": "INFO"
        },
        "file_service": {
            "class": "logging.FileHandler",
            "filename": "logs/service.log",
            "formatter": "default",
            "level": "INFO"
        },
        "file_api": {
            "class": "logging.FileHandler",
            "filename": "logs/api.log",
            "formatter": "default",
            "level": "INFO"
        }
    },
    "loggers": {
        "database": {
            "handlers": ["console", "file_database"],
            "level": "INFO",
            "propagate": False
        },
        "service": {
            "handlers": ["console", "file_service"],
            "level": "INFO",
            "propagate": False
        },
        "api": {
            "handlers": ["console", "file_api"],
            "level": "INFO",
            "propagate": False
        }
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING"
    }
}

environment = os.getenv("ENV", "development")

if environment == "production":
    LOGGING_CONFIG["handlers"]["console"]["level"] = "WARNING"

logging.config.dictConfig(LOGGING_CONFIG)

logging.getLogger("root").info(f"Logging initialized in '{environment}' mode.")
