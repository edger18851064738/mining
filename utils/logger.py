"""
Logging system for the mining dispatch system.

Provides a centralized, configurable logging system that:
- Supports multiple outputs (console, file, etc.)
- Supports different log levels for different components
- Provides convenient log formatting
- Tracks performance metrics
"""

import os
import sys
import time
import logging
import logging.handlers
import threading
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import functools
import traceback
import atexit
from datetime import datetime

# Import the configuration manager
from utils.config import config_manager, LoggingConfig


class LogLevel(Enum):
    """Log levels with corresponding logging module constants."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogFormatter(logging.Formatter):
    """
    Custom log formatter with color support for console output.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def __init__(self, fmt: str = None, datefmt: str = None, use_colors: bool = True):
        """
        Initialize the formatter.
        
        Args:
            fmt: Log message format
            datefmt: Date format
            use_colors: Whether to use colors in console output
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with optional colors."""
        # Save original values
        levelname = record.levelname
        
        # Apply colors if enabled
        if self.use_colors and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the record
        result = super().format(record)
        
        # Restore original values
        record.levelname = levelname
        
        return result


class PerformanceTracker:
    """
    Track performance metrics for different operations.
    """
    
    def __init__(self):
        """Initialize the performance tracker."""
        self.metrics = {}
        self.lock = threading.RLock()
    
    def start_timer(self, operation: str) -> int:
        """
        Start a timer for an operation and return a unique timer ID.
        
        Args:
            operation: Name of the operation
            
        Returns:
            int: Timer ID
        """
        timer_id = id(threading.current_thread()) + int(time.time() * 1000)
        with self.lock:
            if operation not in self.metrics:
                self.metrics[operation] = {
                    'count': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'ongoing': {}
                }
            
            self.metrics[operation]['ongoing'][timer_id] = time.time()
        
        return timer_id
    
    def stop_timer(self, operation: str, timer_id: int) -> float:
        """
        Stop a timer and update metrics.
        
        Args:
            operation: Name of the operation
            timer_id: Timer ID from start_timer
            
        Returns:
            float: Elapsed time in seconds
            
        Raises:
            ValueError: If the timer doesn't exist
        """
        end_time = time.time()
        
        with self.lock:
            if operation not in self.metrics or timer_id not in self.metrics[operation]['ongoing']:
                raise ValueError(f"No timer found for {operation} with ID {timer_id}")
            
            start_time = self.metrics[operation]['ongoing'][timer_id]
            elapsed = end_time - start_time
            
            # Update metrics
            self.metrics[operation]['count'] += 1
            self.metrics[operation]['total_time'] += elapsed
            self.metrics[operation]['min_time'] = min(self.metrics[operation]['min_time'], elapsed)
            self.metrics[operation]['max_time'] = max(self.metrics[operation]['max_time'], elapsed)
            
            # Remove ongoing timer
            del self.metrics[operation]['ongoing'][timer_id]
        
        return elapsed
    
    def get_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Args:
            operation: Operation name, or None for all operations
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        with self.lock:
            if operation:
                # Return metrics for a specific operation
                if operation not in self.metrics:
                    return {}
                
                metrics = dict(self.metrics[operation])
                
                # Add average time
                if metrics['count'] > 0:
                    metrics['avg_time'] = metrics['total_time'] / metrics['count']
                else:
                    metrics['avg_time'] = 0.0
                
                # Remove ongoing timers from the result
                metrics.pop('ongoing', None)
                
                return metrics
            else:
                # Return metrics for all operations
                result = {}
                for op, metrics in self.metrics.items():
                    op_metrics = dict(metrics)
                    
                    # Add average time
                    if op_metrics['count'] > 0:
                        op_metrics['avg_time'] = op_metrics['total_time'] / op_metrics['count']
                    else:
                        op_metrics['avg_time'] = 0.0
                    
                    # Remove ongoing timers from the result
                    op_metrics.pop('ongoing', None)
                    
                    result[op] = op_metrics
                
                return result
    
    def reset(self, operation: Optional[str] = None) -> None:
        """
        Reset performance metrics.
        
        Args:
            operation: Operation name, or None for all operations
        """
        with self.lock:
            if operation:
                # Reset a specific operation
                if operation in self.metrics:
                    ongoing = self.metrics[operation]['ongoing']
                    self.metrics[operation] = {
                        'count': 0,
                        'total_time': 0.0,
                        'min_time': float('inf'),
                        'max_time': 0.0,
                        'ongoing': ongoing
                    }
            else:
                # Reset all operations
                for op in self.metrics:
                    ongoing = self.metrics[op]['ongoing']
                    self.metrics[op] = {
                        'count': 0,
                        'total_time': 0.0,
                        'min_time': float('inf'),
                        'max_time': 0.0,
                        'ongoing': ongoing
                    }


class LogManager:
    """
    Centralized logging manager for the mining dispatch system.
    
    Configures and provides access to loggers for different components,
    handles log rotation, formatting, and output to multiple destinations.
    """
    
    # Singleton instance
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        """
        Initialize the logging manager.
        
        Args:
            config: Logging configuration, or None to use config from ConfigManager
        """
        with self._lock:
            if self._initialized:
                return
            
            # Get logging configuration
            self._config = config or config_manager.get_logging_config()
            
            # Set up the root logger
            self._root_logger = logging.getLogger()
            self._root_logger.setLevel(logging.DEBUG)  # Capture all logs
            
            # Clear existing handlers
            self._root_logger.handlers = []
            
            # Set up console handler
            if self._config.console_output:
                self._setup_console_handler()
            
            # Set up file handler
            if self._config.file_output:
                self._setup_file_handler()
            
            # Create a mapping of logger names to loggers
            self._loggers = {}
            
            # Create performance tracker
            self._performance_tracker = PerformanceTracker()
            
            # Create context for log tracking
            self._context = threading.local()
            self._context.tracking = {}
            
            # Register configuration listener
            config_manager.add_listener(self._on_config_changed)
            
            # Register cleanup on exit
            atexit.register(self._cleanup)
            
            self._initialized = True
            
            # Log initialization complete
            self.get_logger("system").info("Logging system initialized")
    
    def _setup_console_handler(self) -> None:
        """Set up console log handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Use colored formatter for console
        formatter = LogFormatter(
            fmt=self._config.format,
            datefmt=self._config.date_format,
            use_colors=True
        )
        console_handler.setFormatter(formatter)
        
        # Set level from config
        level = getattr(logging, self._config.level, logging.INFO)
        console_handler.setLevel(level)
        
        self._root_logger.addHandler(console_handler)
    
    def _setup_file_handler(self) -> None:
        """Set up file log handler with rotation."""
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(self._config.file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            self._config.file_path,
            maxBytes=self._config.max_file_size,
            backupCount=self._config.backup_count
        )
        
        # Use standard formatter for file
        formatter = logging.Formatter(
            fmt=self._config.format,
            datefmt=self._config.date_format
        )
        file_handler.setFormatter(formatter)
        
        # Set level from config
        level = getattr(logging, self._config.level, logging.INFO)
        file_handler.setLevel(level)
        
        self._root_logger.addHandler(file_handler)
    
    def _on_config_changed(self, section: str) -> None:
        """Handle configuration changes."""
        if section == 'logging':
            # Get updated config
            new_config = config_manager.get_logging_config()
            
            # Update and reconfigure
            self._config = new_config
            self._reconfigure_logging()
    
    def _reconfigure_logging(self) -> None:
        """Reconfigure logging based on current config."""
        # Clear existing handlers
        self._root_logger.handlers = []
        
        # Set up console handler
        if self._config.console_output:
            self._setup_console_handler()
        
        # Set up file handler
        if self._config.file_output:
            self._setup_file_handler()
        
        # Update component log levels
        for name, logger in self._loggers.items():
            level_name = self._config.component_levels.get(name, self._config.level)
            level = getattr(logging, level_name, logging.INFO)
            logger.setLevel(level)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger for a specific component.
        
        Args:
            name: Logger name (typically component name)
            
        Returns:
            logging.Logger: Logger for the component
        """
        with self._lock:
            if name not in self._loggers:
                # Create logger
                logger = logging.getLogger(name)
                
                # Set level from component config
                level_name = self._config.component_levels.get(name, self._config.level)
                level = getattr(logging, level_name, logging.INFO)
                logger.setLevel(level)
                
                # Store for later access
                self._loggers[name] = logger
            
            return self._loggers[name]
    
    def start_timer(self, operation: str) -> int:
        """
        Start a timer for performance tracking.
        
        Args:
            operation: Name of the operation
            
        Returns:
            int: Timer ID
        """
        return self._performance_tracker.start_timer(operation)
    
    def stop_timer(self, operation: str, timer_id: int) -> float:
        """
        Stop a timer and return elapsed time.
        
        Args:
            operation: Name of the operation
            timer_id: Timer ID from start_timer
            
        Returns:
            float: Elapsed time in seconds
        """
        return self._performance_tracker.stop_timer(operation, timer_id)
    
    def get_performance_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Args:
            operation: Operation name, or None for all operations
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        return self._performance_tracker.get_metrics(operation)
    
    def reset_performance_metrics(self, operation: Optional[str] = None) -> None:
        """
        Reset performance metrics.
        
        Args:
            operation: Operation name, or None for all operations
        """
        self._performance_tracker.reset(operation)
    
    def _cleanup(self) -> None:
        """Clean up resources when the application exits."""
        # Flush all log handlers
        for handler in self._root_logger.handlers:
            handler.flush()
            handler.close()
        
        # Log final shutdown
        try:
            for handler in self._root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    # For stream handlers, use the stream directly
                    handler.stream.write("[SHUTDOWN] Logging system shutting down\n")
                    handler.stream.flush()
                elif isinstance(handler, logging.FileHandler):
                    # For file handlers, open the file directly
                    with open(handler.baseFilename, 'a') as f:
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - [SHUTDOWN] Logging system shutting down\n")
        except Exception:
            pass  # Ignore errors during cleanup


# Performance tracking decorator
def timed(operation: str):
    """
    Decorator to track execution time of a function.
    
    Args:
        operation: Name of the operation for tracking
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_manager = LogManager()
            timer_id = log_manager.start_timer(operation)
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = log_manager.stop_timer(operation, timer_id)
                log_manager.get_logger("performance").debug(
                    f"{operation} completed in {elapsed:.3f} seconds"
                )
        return wrapper
    return decorator


# Exception logging decorator
def log_exceptions(logger_name: str = "exceptions"):
    """
    Decorator to log exceptions raised by a function.
    
    Args:
        logger_name: Name of the logger to use
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get the logger
                logger = LogManager().get_logger(logger_name)
                
                # Log the exception
                logger.error(
                    f"Exception in {func.__name__}: {str(e)}\n"
                    f"{traceback.format_exc()}"
                )
                
                # Re-raise the exception
                raise
        return wrapper
    return decorator


# Create global log manager instance
log_manager = LogManager()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific component.
    
    Args:
        name: Logger name (typically component name)
        
    Returns:
        logging.Logger: Logger for the component
    """
    return log_manager.get_logger(name)


# Performance tracking functions
def start_timer(operation: str) -> int:
    """
    Start a timer for performance tracking.
    
    Args:
        operation: Name of the operation
        
    Returns:
        int: Timer ID
    """
    return log_manager.start_timer(operation)


def stop_timer(operation: str, timer_id: int) -> float:
    """
    Stop a timer and return elapsed time.
    
    Args:
        operation: Name of the operation
        timer_id: Timer ID from start_timer
        
    Returns:
        float: Elapsed time in seconds
    """
    return log_manager.stop_timer(operation, timer_id)


def get_performance_metrics(operation: Optional[str] = None) -> Dict[str, Any]:
    """
    Get performance metrics.
    
    Args:
        operation: Operation name, or None for all operations
        
    Returns:
        Dict[str, Any]: Performance metrics
    """
    return log_manager.get_performance_metrics(operation)


def reset_performance_metrics(operation: Optional[str] = None) -> None:
    """
    Reset performance metrics.
    
    Args:
        operation: Operation name, or None for all operations
    """
    log_manager.reset_performance_metrics(operation)