"""
app/utils/logging_utils.py

Configures structured (JSON) logging for the application.
"""

import logging
import sys
import json
from app.config import settings

# Custom Formatter for JSON logging
class JsonFormatter(logging.Formatter):
    """
    Formats log records as a JSON string for consumption by
    log management systems (e.g., Splunk, ELK).
    """
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_record)

def setup_logging() -> logging.Logger:
    """
    Configures and returns the root logger for the application.
    This function is called once at startup from main.py.
    """
    # Get the root logger
    logger = logging.getLogger()
    
    # Set log level from config
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create stream handler (stdout)
    handler = logging.StreamHandler(sys.stdout)

    if settings.JSON_LOGS:
        # Use JSON formatter for production
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
    else:
        # Use a simple, readable formatter for local dev
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # Silence noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    root_logger = logging.getLogger(__name__)
    root_logger.info(
        f"Logging configured. Level: {settings.LOG_LEVEL}. "
        f"JSON: {settings.JSON_LOGS}."
    )
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """Helper function to get a logger for a specific module."""
    return logging.getLogger(name)