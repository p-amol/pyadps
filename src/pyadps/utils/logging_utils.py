"""
Reusable Logging Utilities
A clean, configurable logging module that can be used across multiple projects.
"""

import logging
import sys
from enum import Enum
from typing import Optional, Union
from pathlib import Path


class LogLevel(Enum):
    """Log level enumeration"""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class CustomFormatter(logging.Formatter):
    """Custom colored formatter for console logging"""

    COLORS = {
        logging.DEBUG: "\x1b[36m",  # Cyan
        logging.INFO: "\x1b[32m",  # Green
        logging.WARNING: "\x1b[33m",  # Yellow
        logging.ERROR: "\x1b[31m",  # Red
        logging.CRITICAL: "\x1b[31;1m",  # Bold Red
    }
    RESET = "\x1b[0m"

    def __init__(self, include_timestamp: bool = True, include_module: bool = False):
        """
        Initialize formatter with optional components

        Args:
            include_timestamp: Whether to include timestamp in log format
            include_module: Whether to include module name in log format
        """
        self.include_timestamp = include_timestamp
        self.include_module = include_module
        super().__init__()

    def format(self, record):
        """Format log record with colors and optional components"""
        # Build format string based on options
        format_parts = []

        if self.include_timestamp:
            format_parts.append("%(asctime)s")

        format_parts.append("%(levelname)s")

        if self.include_module:
            format_parts.append("%(name)s")

        format_parts.append("%(message)s")

        log_format = " - ".join(format_parts)

        # Apply color
        color = self.COLORS.get(record.levelno, "")
        colored_format = color + log_format + self.RESET

        formatter = logging.Formatter(
            colored_format,
            datefmt="%Y-%m-%d %H:%M:%S" if self.include_timestamp else None,
        )
        return formatter.format(record)


class LoggerConfig:
    """Configuration class for logger setup"""

    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        include_timestamp: bool = True,
        include_module: bool = False,
        log_to_file: bool = False,
        log_file_path: Optional[Union[str, Path]] = None,
        file_log_level: Optional[LogLevel] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ):
        """
        Initialize logger configuration

        Args:
            level: Console logging level
            include_timestamp: Include timestamp in console output
            include_module: Include module name in output
            log_to_file: Whether to also log to file
            log_file_path: Path for log file (if log_to_file is True)
            file_log_level: File logging level (defaults to console level)
            max_file_size: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
        """
        self.level = level
        self.include_timestamp = include_timestamp
        self.include_module = include_module
        self.log_to_file = log_to_file
        self.log_file_path = Path(log_file_path) if log_file_path else None
        self.file_log_level = file_log_level or level
        self.max_file_size = max_file_size
        self.backup_count = backup_count


class LoggerManager:
    """Manages logger configuration and setup"""

    _loggers = {}  # Cache for created loggers

    @classmethod
    def setup_logger(
        self, name: str = "app", config: Optional[LoggerConfig] = None
    ) -> logging.Logger:
        """
        Set up and configure logger with given configuration

        Args:
            name: Logger name
            config: Logger configuration (uses defaults if None)

        Returns:
            Configured logger instance
        """
        # Use default config if none provided
        if config is None:
            config = LoggerConfig()

        # Return cached logger if it exists
        cache_key = f"{name}_{id(config)}"
        if cache_key in self._loggers:
            return self._loggers[cache_key]

        logger = logging.getLogger(name)
        logger.setLevel(config.level.value)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level.value)
        console_formatter = CustomFormatter(
            include_timestamp=config.include_timestamp,
            include_module=config.include_module,
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # Add file handler if requested
        if config.log_to_file and config.log_file_path:
            self._add_file_handler(logger, config)

        # Prevent propagation to root logger
        logger.propagate = False

        # Cache the logger
        self._loggers[cache_key] = logger

        return logger

    @classmethod
    def _add_file_handler(self, logger: logging.Logger, config: LoggerConfig):
        """Add rotating file handler to logger"""
        from logging.handlers import RotatingFileHandler

        # Ensure log directory exists
        config.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            config.log_file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
        )
        file_handler.setLevel(config.file_log_level.value)

        # File logs typically include more detail
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    @classmethod
    def get_logger(self, name: str = "app") -> logging.Logger:
        """Get existing logger or create with default config"""
        return logging.getLogger(name) or self.setup_logger(name)

    @classmethod
    def clear_cache(self):
        """Clear logger cache (useful for testing)"""
        self._loggers.clear()


# Convenience functions for quick setup
def get_console_logger(
    name: str = "app", level: LogLevel = LogLevel.INFO, include_timestamp: bool = True
) -> logging.Logger:
    """Quick setup for console-only logger"""
    config = LoggerConfig(
        level=level, include_timestamp=include_timestamp, include_module=False
    )
    return LoggerManager.setup_logger(name, config)


def get_file_logger(
    name: str = "app",
    log_file: Union[str, Path] = "app.log",
    level: LogLevel = LogLevel.INFO,
    file_level: Optional[LogLevel] = None,
) -> logging.Logger:
    """Quick setup for file + console logger"""
    config = LoggerConfig(
        level=level,
        log_to_file=True,
        log_file_path=log_file,
        file_log_level=file_level or LogLevel.DEBUG,
    )
    return LoggerManager.setup_logger(name, config)


def get_detailed_logger(
    name: str = "app",
    log_file: Union[str, Path] = "app.log",
    console_level: LogLevel = LogLevel.INFO,
    file_level: LogLevel = LogLevel.DEBUG,
) -> logging.Logger:
    """Setup logger with detailed configuration"""
    config = LoggerConfig(
        level=console_level,
        include_timestamp=True,
        include_module=True,
        log_to_file=True,
        log_file_path=log_file,
        file_log_level=file_level,
    )
    return LoggerManager.setup_logger(name, config)


# Example usage
if __name__ == "__main__":
    # Test different logger configurations

    # Simple console logger
    simple_logger = get_console_logger("simple", LogLevel.DEBUG)
    simple_logger.debug("Debug message")
    simple_logger.info("Info message")
    simple_logger.warning("Warning message")
    simple_logger.error("Error message")

    print("\n" + "=" * 50 + "\n")

    # File + console logger
    file_logger = get_file_logger("file_test", "test.log", LogLevel.INFO)
    file_logger.info("This goes to both console and file")
    file_logger.debug("This only goes to file")

    print("\n" + "=" * 50 + "\n")

    # Detailed logger
    detailed_logger = get_detailed_logger("detailed", "detailed.log")
    detailed_logger.info("Detailed logging with module names")
