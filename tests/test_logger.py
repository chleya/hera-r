# -*- coding: utf-8 -*-
"""Tests for logging module."""

import io
import sys
from unittest.mock import patch
from hera.utils.logger import HeraLogger, console


def test_logger_initialization():
    """Test logger initialization with different configs."""
    # Test with verbose enabled
    config_verbose = {"logging": {"verbose": True}}
    logger = HeraLogger(config_verbose)
    assert logger.verbose is True
    
    # Test with verbose disabled
    config_quiet = {"logging": {"verbose": False}}
    logger = HeraLogger(config_quiet)
    assert logger.verbose is False
    
    # Test with missing logging config
    config_minimal = {}
    logger = HeraLogger(config_minimal)
    assert logger.verbose is True  # Default should be True


def test_logger_methods():
    """Test all logger methods."""
    config = {"logging": {"verbose": True}}
    logger = HeraLogger(config)
    
    # Capture output
    with io.StringIO() as buf, patch('sys.stdout', buf):
        # Test info method (should print when verbose=True)
        logger.info("Test info message")
        output = buf.getvalue()
        assert "Test info message" in output
        assert "ℹ️" in output
    
    with io.StringIO() as buf, patch('sys.stdout', buf):
        # Test step method (always prints)
        logger.step("Test step message")
        output = buf.getvalue()
        assert "Test step message" in output
        assert "[REFRESH]" in output
    
    with io.StringIO() as buf, patch('sys.stdout', buf):
        # Test success method (always prints)
        logger.success("Test success message")
        output = buf.getvalue()
        assert "Test success message" in output
        assert "[OK]" in output
    
    with io.StringIO() as buf, patch('sys.stdout', buf):
        # Test warning method (always prints)
        logger.warning("Test warning message")
        output = buf.getvalue()
        assert "Test warning message" in output
        assert "[WARN]" in output
    
    with io.StringIO() as buf, patch('sys.stdout', buf):
        # Test error method (always prints)
        logger.error("Test error message")
        output = buf.getvalue()
        assert "Test error message" in output
        assert "[FAIL]" in output


def test_logger_verbose_control():
    """Test that info messages respect verbose setting."""
    # Test with verbose=False
    config_quiet = {"logging": {"verbose": False}}
    logger = HeraLogger(config_quiet)
    
    with io.StringIO() as buf, patch('sys.stdout', buf):
        logger.info("This should not appear")
        output = buf.getvalue()
        assert output == ""  # Should be empty when verbose=False
    
    # Test with verbose=True
    config_verbose = {"logging": {"verbose": True}}
    logger = HeraLogger(config_verbose)
    
    with io.StringIO() as buf, patch('sys.stdout', buf):
        logger.info("This should appear")
        output = buf.getvalue()
        assert "This should appear" in output


def test_logger_timestamp_format():
    """Test that timestamps are included in output."""
    config = {"logging": {"verbose": True}}
    logger = HeraLogger(config)
    
    with io.StringIO() as buf, patch('sys.stdout', buf):
        logger.info("Test message")
        output = buf.getvalue()
        
        # Check timestamp format (HH:MM:SS)
        import re
        timestamp_pattern = r"\d{2}:\d{2}:\d{2}"
        assert re.search(timestamp_pattern, output) is not None


def test_console_object():
    """Test that console object is properly configured."""
    # Check console theme
    assert hasattr(console, "theme")
    assert console.theme is not None
    
    # Check theme colors
    theme_colors = ["info", "warning", "error", "success", "step"]
    for color in theme_colors:
        assert color in console.theme.styles


def test_logger_edge_cases():
    """Test logger with edge cases."""
    config = {"logging": {"verbose": True}}
    logger = HeraLogger(config)
    
    # Test empty message
    with io.StringIO() as buf, patch('sys.stdout', buf):
        logger.info("")
        output = buf.getvalue()
        assert "ℹ️" in output
    
    # Test very long message
    long_message = "A" * 1000
    with io.StringIO() as buf, patch('sys.stdout', buf):
        logger.info(long_message)
        output = buf.getvalue()
        assert "A" * 100 in output  # At least part of the message should appear
    
    # Test special characters
    special_message = "Test with special chars: \n\t\"'&<>"
    with io.StringIO() as buf, patch('sys.stdout', buf):
        logger.info(special_message)
        output = buf.getvalue()
        assert "Test with special chars" in output


def test_logger_config_handling():
    """Test logger handles various config structures."""
    test_cases = [
        # (config, expected_verbose)
        ({}, True),  # Default
        ({"logging": {}}, True),  # Empty logging section
        ({"logging": {"verbose": True}}, True),
        ({"logging": {"verbose": False}}, False),
        ({"logging": {"verbose": "true"}}, True),  # String true
        ({"logging": {"verbose": "false"}}, True),  # String false (non-boolean)
        ({"logging": {"other_setting": "value"}}, True),  # Missing verbose
    ]
    
    for config, expected_verbose in test_cases:
        logger = HeraLogger(config)
        assert logger.verbose == expected_verbose, f"Failed for config: {config}"