"""Logging configuration for MLOps pipeline."""

import logging
import sys


def setup_logging(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the MLOps pipeline.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout
    )
    
    logger = logging.getLogger(name)
    return logger
