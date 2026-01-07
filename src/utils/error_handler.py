"""
Error handling utilities with circuit breaker and graceful degradation.

Provides robust error handling, user-friendly error messages, and circuit breaker
pattern for LLM calls.
"""

import time
from typing import Optional, Callable, Any, Dict
from enum import Enum
from functools import wraps

from src.utils.logger import get_logger

logger = get_logger()


class ErrorType(Enum):
    """Error type enumeration."""

    VALIDATION_ERROR = "validation_error"
    SQL_GENERATION_ERROR = "sql_generation_error"
    DATABASE_ERROR = "database_error"
    LLM_ERROR = "llm_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """Circuit breaker pattern implementation for LLM calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        if self.state == "open":
            if time.time() - (self.last_failure_time or 0) >= self.recovery_timeout:
                self.state = "half_open"
                logger.info("Circuit breaker transitioning to half-open state")
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker is OPEN. Too many failures. "
                    f"Retry after {self.recovery_timeout} seconds."
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        if self.state == "half_open":
            logger.info("Circuit breaker closed after successful call")
            self.state = "closed"
        self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures"
            )


class ErrorHandler:
    """Error handler with user-friendly error messages."""

    ERROR_MESSAGES: Dict[ErrorType, str] = {
        ErrorType.VALIDATION_ERROR: "The generated SQL query failed validation checks. Please try rephrasing your question.",
        ErrorType.SQL_GENERATION_ERROR: "Unable to generate SQL query from your question. Please try rephrasing or simplifying your request.",
        ErrorType.DATABASE_ERROR: "Database connection error. Please try again later.",
        ErrorType.LLM_ERROR: "AI service temporarily unavailable. Please try again in a moment.",
        ErrorType.NETWORK_ERROR: "Network error occurred. Please check your connection and try again.",
        ErrorType.TIMEOUT_ERROR: "Request timed out. Please try again with a simpler question.",
        ErrorType.UNKNOWN_ERROR: "An unexpected error occurred. Please try again or contact support.",
    }

    @staticmethod
    def get_user_friendly_message(error: Exception, error_type: ErrorType) -> str:
        """
        Get user-friendly error message.

        Args:
            error: The exception that occurred
            error_type: Type of error

        Returns:
            User-friendly error message
        """
        base_message = ErrorHandler.ERROR_MESSAGES.get(
            error_type, ErrorHandler.ERROR_MESSAGES[ErrorType.UNKNOWN_ERROR]
        )

        # Add specific details for certain error types
        if error_type == ErrorType.VALIDATION_ERROR:
            error_str = str(error)
            if "forbidden" in error_str.lower():
                base_message += " The query contains forbidden operations."
            elif "syntax" in error_str.lower():
                base_message += " There was a syntax error in the generated query."

        return base_message

    @staticmethod
    def classify_error(error: Exception) -> ErrorType:
        """
        Classify error type.

        Args:
            error: Exception to classify

        Returns:
            ErrorType enum value
        """
        error_str = str(error).lower()
        error_name = type(error).__name__.lower()

        if "validation" in error_name or "validation" in error_str:
            return ErrorType.VALIDATION_ERROR
        elif "sql" in error_name or "query" in error_str:
            return ErrorType.SQL_GENERATION_ERROR
        elif "database" in error_name or "connection" in error_str or "psycopg" in error_str:
            return ErrorType.DATABASE_ERROR
        elif "llm" in error_name or "openai" in error_str or "gemini" in error_str or "api" in error_str:
            return ErrorType.LLM_ERROR
        elif "timeout" in error_name or "timeout" in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif "network" in error_name or "connection" in error_str:
            return ErrorType.NETWORK_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR


def graceful_degradation(fallback_value: Any = None):
    """
    Decorator for graceful degradation.

    Args:
        fallback_value: Value to return on failure

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Function {func.__name__} failed, using fallback",
                    error=str(e),
                )
                return fallback_value

        return wrapper

    return decorator


# Global circuit breaker instance for LLM calls
_llm_circuit_breaker: Optional[CircuitBreaker] = None


def get_llm_circuit_breaker() -> CircuitBreaker:
    """
    Get or create global LLM circuit breaker.

    Returns:
        CircuitBreaker instance
    """
    global _llm_circuit_breaker
    if _llm_circuit_breaker is None:
        _llm_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=Exception,
        )
    return _llm_circuit_breaker

