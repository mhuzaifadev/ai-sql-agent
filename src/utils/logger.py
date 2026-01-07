"""
Structured logging and monitoring utilities.

Provides logging, metrics tracking, and monitoring capabilities for the SQL Compiler Agent.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps
from contextlib import contextmanager
import os


class StructuredLogger:
    """Structured logger with JSON output support."""

    def __init__(
        self,
        name: str = "sql_compiler_agent",
        log_level: str = "INFO",
        json_output: bool = False,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            json_output: Whether to output logs in JSON format
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.json_output = json_output

        # Remove existing handlers
        self.logger.handlers.clear()

        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, log_level.upper()))

        if json_output:
            formatter = logging.Formatter("%(message)s")
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs,
        }

        if self.json_output:
            log_message = json.dumps(log_data)
        else:
            # Format as readable string
            extra_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            log_message = f"{message}" + (f" | {extra_str}" if extra_str else "")

        getattr(self.logger, level.lower())(log_message)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log("CRITICAL", message, **kwargs)


class MetricsCollector:
    """Collects and tracks metrics for the SQL Compiler Agent."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Any] = {
            "query_executions": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "retry_counts": [],
            "llm_calls": 0,
            "llm_tokens_used": 0,
            "average_response_time": 0.0,
            "response_times": [],
            "errors_by_type": {},
        }

    def record_query_execution(
        self,
        success: bool,
        retries: int = 0,
        response_time: float = 0.0,
        error_type: Optional[str] = None,
    ):
        """
        Record a query execution.

        Args:
            success: Whether the query executed successfully
            retries: Number of retries attempted
            response_time: Response time in seconds
            error_type: Type of error if failed
        """
        self.metrics["query_executions"] += 1

        if success:
            self.metrics["successful_queries"] += 1
        else:
            self.metrics["failed_queries"] += 1
            if error_type:
                self.metrics["errors_by_type"][error_type] = (
                    self.metrics["errors_by_type"].get(error_type, 0) + 1
                )

        if retries > 0:
            self.metrics["retry_counts"].append(retries)

        if response_time > 0:
            self.metrics["response_times"].append(response_time)
            # Keep only last 1000 response times
            if len(self.metrics["response_times"]) > 1000:
                self.metrics["response_times"] = self.metrics["response_times"][-1000:]

            # Calculate average
            self.metrics["average_response_time"] = sum(
                self.metrics["response_times"]
            ) / len(self.metrics["response_times"])

    def record_llm_call(self, tokens_used: int = 0):
        """
        Record an LLM API call.

        Args:
            tokens_used: Number of tokens used in the call
        """
        self.metrics["llm_calls"] += 1
        if tokens_used > 0:
            self.metrics["llm_tokens_used"] += tokens_used

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.

        Returns:
            Dictionary of current metrics
        """
        metrics = self.metrics.copy()

        # Calculate success rate
        if metrics["query_executions"] > 0:
            metrics["success_rate"] = (
                metrics["successful_queries"] / metrics["query_executions"]
            ) * 100
        else:
            metrics["success_rate"] = 0.0

        # Calculate average retries
        if metrics["retry_counts"]:
            metrics["average_retries"] = sum(metrics["retry_counts"]) / len(
                metrics["retry_counts"]
            )
        else:
            metrics["average_retries"] = 0.0

        # Calculate average tokens per call
        if metrics["llm_calls"] > 0:
            metrics["average_tokens_per_call"] = (
                metrics["llm_tokens_used"] / metrics["llm_calls"]
            )
        else:
            metrics["average_tokens_per_call"] = 0.0

        return metrics

    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            "query_executions": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "retry_counts": [],
            "llm_calls": 0,
            "llm_tokens_used": 0,
            "average_response_time": 0.0,
            "response_times": [],
            "errors_by_type": {},
        }


# Global logger and metrics instances
_logger_instance: Optional[StructuredLogger] = None
_metrics_instance: Optional[MetricsCollector] = None


def get_logger() -> StructuredLogger:
    """
    Get or create global logger instance.

    Returns:
        StructuredLogger instance
    """
    global _logger_instance
    if _logger_instance is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
        json_output = os.getenv("JSON_LOGS", "false").lower() == "true"
        _logger_instance = StructuredLogger(
            log_level=log_level, json_output=json_output
        )
    return _logger_instance


def get_metrics() -> MetricsCollector:
    """
    Get or create global metrics collector instance.

    Returns:
        MetricsCollector instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


@contextmanager
def track_execution_time(operation: str):
    """
    Context manager to track execution time of an operation.

    Args:
        operation: Name of the operation being tracked

    Yields:
        None
    """
    start_time = time.time()
    logger = get_logger()
    logger.debug(f"Starting {operation}")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.debug(f"Completed {operation}", duration_seconds=elapsed)


def log_query_execution(func):
    """
    Decorator to log query execution metrics.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = time.time()
        success = False
        retries = 0
        error_type = None

        try:
            result = func(*args, **kwargs)
            success = True
            return result
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Query execution failed: {str(e)}", error_type=error_type)
            raise
        finally:
            elapsed = time.time() - start_time
            metrics = get_metrics()
            metrics.record_query_execution(
                success=success,
                retries=retries,
                response_time=elapsed,
                error_type=error_type,
            )

    return wrapper

