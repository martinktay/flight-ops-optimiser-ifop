"""
Logging utility for the flight operations optimisation project.

This module provides a centralised logging configuration with structured
logging capabilities and proper formatting for the project.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

import structlog


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    structured: bool = True
) -> None:
    """
    Set up logging configuration for the project.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output logs to console
        structured: Whether to use structured logging
    """
    # Create logs directory if log file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    if structured:
        _setup_structured_logging(level, log_file, console_output)
    else:
        _setup_standard_logging(level, log_file, console_output)


def _setup_structured_logging(
    level: str,
    log_file: Optional[str],
    console_output: bool
) -> None:
    """
    Set up structured logging using structlog.

    Args:
        level: Logging level
        log_file: Path to log file
        console_output: Whether to output to console
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set up handlers
    handlers = []

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        handlers.append(console_handler)

    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        handlers=handlers
    )


def _setup_standard_logging(
    level: str,
    log_file: Optional[str],
    console_output: bool
) -> None:
    """
    Set up standard Python logging.

    Args:
        level: Logging level
        log_file: Path to log file
        console_output: Whether to output to console
    """
    # Define log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Set up handlers
    handlers = []

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(log_format, date_format)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def get_structured_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance for the specified module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        structlog.BoundLogger: Configured structured logger instance
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """
    A mixin class that provides logging capabilities to other classes.

    This mixin automatically creates a logger instance for the class
    and provides convenient logging methods.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the logger mixin."""
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)

    def log_info(self, message: str, **kwargs) -> None:
        """Log an info message with optional context."""
        if kwargs:
            self.logger.info(f"{message} - {kwargs}")
        else:
            self.logger.info(message)

    def log_warning(self, message: str, **kwargs) -> None:
        """Log a warning message with optional context."""
        if kwargs:
            self.logger.warning(f"{message} - {kwargs}")
        else:
            self.logger.warning(message)

    def log_error(self, message: str, **kwargs) -> None:
        """Log an error message with optional context."""
        if kwargs:
            self.logger.error(f"{message} - {kwargs}")
        else:
            self.logger.error(message)

    def log_debug(self, message: str, **kwargs) -> None:
        """Log a debug message with optional context."""
        if kwargs:
            self.logger.debug(f"{message} - {kwargs}")
        else:
            self.logger.debug(message)

    def log_exception(self, message: str, exc_info: bool = True) -> None:
        """Log an exception with traceback."""
        self.logger.exception(message, exc_info=exc_info)


class PerformanceLogger:
    """
    A specialised logger for tracking performance metrics and timing.

    This class provides methods for logging performance-related information
    such as execution times, memory usage, and other metrics.
    """

    def __init__(self, name: str = "PerformanceLogger") -> None:
        """
        Initialise the performance logger.

        Args:
            name: Logger name
        """
        self.logger = get_logger(name)
        self.timers = {}

    def start_timer(self, operation: str) -> None:
        """
        Start timing an operation.

        Args:
            operation: Name of the operation being timed
        """
        import time
        self.timers[operation] = time.time()
        self.logger.info(f"Started timing operation: {operation}")

    def end_timer(self, operation: str) -> float:
        """
        End timing an operation and log the duration.

        Args:
            operation: Name of the operation being timed

        Returns:
            float: Duration in seconds
        """
        import time
        if operation not in self.timers:
            self.logger.warning(f"No timer found for operation: {operation}")
            return 0.0

        duration = time.time() - self.timers[operation]
        self.logger.info(
            f"Operation '{operation}' completed in {duration:.2f} seconds")
        del self.timers[operation]
        return duration

    def log_memory_usage(self, operation: str = "Memory Usage") -> None:
        """
        Log current memory usage.

        Args:
            operation: Description of the operation
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.logger.info(f"{operation}: {memory_mb:.2f} MB")
        except ImportError:
            self.logger.warning(
                "psutil not available, cannot log memory usage")

    def log_performance_metrics(self, metrics: dict, operation: str = "Performance Metrics") -> None:
        """
        Log performance metrics.

        Args:
            metrics: Dictionary of performance metrics
            operation: Description of the operation
        """
        self.logger.info(f"{operation}: {metrics}")

    def log_model_performance(self, model_name: str, metrics: dict) -> None:
        """
        Log machine learning model performance metrics.

        Args:
            model_name: Name of the model
            metrics: Dictionary of model metrics
        """
        self.logger.info(f"Model '{model_name}' performance: {metrics}")

    def log_optimisation_performance(self, solver: str, metrics: dict) -> None:
        """
        Log optimisation solver performance metrics.

        Args:
            solver: Name of the optimisation solver
            metrics: Dictionary of solver metrics
        """
        self.logger.info(
            f"Optimisation solver '{solver}' performance: {metrics}")


# Default logging setup
setup_logging()
