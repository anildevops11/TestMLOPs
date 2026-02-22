"""Test logging configuration."""

import logging
from src.logging_config import setup_logging


def test_setup_logging():
    """Verify logging configuration works."""
    logger = setup_logging(__name__)
    
    assert logger is not None
    assert isinstance(logger, logging.Logger)
    assert logger.name == __name__


def test_logging_output(caplog):
    """Verify logging produces output."""
    logger = setup_logging(__name__, level=logging.INFO)
    
    with caplog.at_level(logging.INFO):
        logger.info("Test message")
    
    assert "Test message" in caplog.text
