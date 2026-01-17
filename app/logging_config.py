"""
Logging Configuration Module
Centralized logging setup with file rotation and structured logging
"""
import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
import sys
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: blue + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def setup_logging(
    log_level: str = None,
    log_dir: str = "logs",
    console_output: bool = True
):
    """
    Setup application logging with file and console handlers
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        console_output: Whether to output to console
    """
    # Get log level from environment or parameter
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # File handler - rotating by size
    app_log_file = log_path / "app.log"
    file_handler = RotatingFileHandler(
        app_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Daily rotating file handler for archival
    daily_log_file = log_path / "daily.log"
    daily_handler = TimedRotatingFileHandler(
        daily_log_file,
        when='midnight',
        interval=1,
        backupCount=30,  # Keep 30 days
        encoding='utf-8'
    )
    daily_handler.setLevel(numeric_level)
    daily_handler.setFormatter(file_formatter)
    daily_handler.suffix = "%Y-%m-%d"
    root_logger.addHandler(daily_handler)
    
    # Error file handler - only errors and critical
    error_log_file = log_path / "errors.log"
    error_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(ColoredFormatter())
        root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    
    # Reduce verbosity of third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    # Log startup message
    root_logger.info("=" * 60)
    root_logger.info("Logging initialized")
    root_logger.info(f"Log level: {log_level}")
    root_logger.info(f"Log directory: {log_path.absolute()}")
    root_logger.info(f"App log: {app_log_file}")
    root_logger.info(f"Error log: {error_log_file}")
    root_logger.info(f"Daily log: {daily_log_file}")
    root_logger.info("=" * 60)


def get_request_logger():
    """Get logger for API requests"""
    return logging.getLogger("api.requests")


def get_agent_logger():
    """Get logger for agent operations"""
    return logging.getLogger("agent")


def get_database_logger():
    """Get logger for database operations"""
    return logging.getLogger("database")


# Request/Response logging helper
class RequestLogger:
    """Helper class for structured request/response logging"""
    
    def __init__(self):
        self.logger = get_request_logger()
    
    def log_request(self, session_id: str, endpoint: str, data: dict):
        """Log incoming request"""
        self.logger.info(
            f"REQUEST | Session: {session_id} | Endpoint: {endpoint} | "
            f"Data: {self._sanitize_data(data)}"
        )
    
    def log_response(self, session_id: str, endpoint: str, status: int, duration: float):
        """Log response"""
        self.logger.info(
            f"RESPONSE | Session: {session_id} | Endpoint: {endpoint} | "
            f"Status: {status} | Duration: {duration:.3f}s"
        )
    
    def log_error(self, session_id: str, endpoint: str, error: str):
        """Log error"""
        self.logger.error(
            f"ERROR | Session: {session_id} | Endpoint: {endpoint} | "
            f"Error: {error}"
        )
    
    @staticmethod
    def _sanitize_data(data: dict) -> dict:
        """Remove sensitive information from logs"""
        sanitized = data.copy()
        
        # Remove sensitive fields
        sensitive_fields = ['api_key', 'password', 'token', 'secret']
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = "***REDACTED***"
        
        # Limit length
        str_data = str(sanitized)
        if len(str_data) > 500:
            return str_data[:500] + "..."
        
        return sanitized


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="DEBUG")
    
    # Test different log levels
    logger = logging.getLogger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test request logger
    req_logger = RequestLogger()
    req_logger.log_request(
        session_id="test-123",
        endpoint="/chat",
        data={"message": "Hello", "api_key": "secret123"}
    )
    req_logger.log_response(
        session_id="test-123",
        endpoint="/chat",
        status=200,
        duration=1.234
    )
