"""
FastAPI application for SQL Compiler Agent.

Provides REST API interface for natural language to SQL conversion.
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager

from src.pipeline.graph import create_graph, SQLCompilerGraph
from src.database.connection import DatabaseConnection
from src.utils.logger import get_logger, get_metrics, track_execution_time
from src.utils.error_handler import (
    ErrorHandler,
    ErrorType,
    get_llm_circuit_breaker,
    CircuitBreakerError,
)

# Initialize logger and metrics
logger = get_logger()
metrics = get_metrics()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global graph instance
graph_instance: Optional[SQLCompilerGraph] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global graph_instance

    # Startup
    logger.info("Starting SQL Compiler Agent API")
    try:
        db_connection = DatabaseConnection()
        db_connection.connect()
        graph_instance = create_graph(db_connection)
        logger.info("Database connection established and graph initialized")
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down SQL Compiler Agent API")
    if graph_instance and graph_instance.db_connection:
        graph_instance.db_connection.close()


# Create FastAPI app
app = FastAPI(
    title="SQL Compiler Agent API",
    description="Natural Language to SQL conversion API",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for SQL query generation."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural language question to convert to SQL",
    )
    max_retries: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="Maximum number of retry attempts (overrides env var)",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate question is not empty."""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class QueryResponse(BaseModel):
    """Response model for SQL query generation."""

    success: bool
    sql_query: Optional[str] = None
    explanation: Optional[str] = None
    execution_result: Optional[dict] = None
    error: Optional[str] = None
    retries: int = 0
    execution_time: float = 0.0


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    metrics: dict


class MetricsResponse(BaseModel):
    """Metrics response model."""

    metrics: dict


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "SQL Compiler Agent API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status and basic metrics
    """
    try:
        # Check database connection
        if graph_instance and graph_instance.db_connection:
            db_healthy = graph_instance.db_connection.test_connection()
        else:
            db_healthy = False

        status = "healthy" if db_healthy else "degraded"

        return HealthResponse(
            status=status,
            version="1.0.0",
            metrics=metrics.get_metrics(),
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            metrics={},
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics_endpoint():
    """
    Get application metrics.

    Returns:
        Current metrics
    """
    return MetricsResponse(metrics=metrics.get_metrics())


@app.post("/query", response_model=QueryResponse, tags=["Query"])
@limiter.limit(os.getenv("RATE_LIMIT", "10/minute"))
async def generate_sql(
    request: QueryRequest,
    http_request: Request,
):
    """
    Convert natural language question to SQL and execute.

    Args:
        request: Query request with natural language question
        http_request: HTTP request object for rate limiting

    Returns:
        Query response with SQL and execution results
    """
    import time

    start_time = time.time()
    retries = 0

    try:
        logger.info(
            "Received query request",
            question=request.question[:100],
            max_retries=request.max_retries,
        )

        # Check circuit breaker
        circuit_breaker = get_llm_circuit_breaker()
        if circuit_breaker.state == "open":
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. Please try again later.",
            )

        # Temporarily override MAX_RETRIES if provided
        original_max_retries = os.getenv("MAX_RETRIES")
        if request.max_retries is not None:
            os.environ["MAX_RETRIES"] = str(request.max_retries)

        try:
            # Execute graph
            with track_execution_time("query_execution"):
                result = graph_instance.invoke(request.question)

            retries = result.get("retries", 0)
            sql_query = result.get("sql_query")
            execution_result = result.get("execution_result")
            error = result.get("error")

            # Record metrics
            elapsed = time.time() - start_time
            success = execution_result is not None and execution_result.get(
                "success", False
            )
            metrics.record_query_execution(
                success=success,
                retries=retries,
                response_time=elapsed,
                error_type=ErrorHandler.classify_error(
                    Exception(error) if error else Exception()
                ).value
                if error
                else None,
            )

            if error:
                error_type = ErrorHandler.classify_error(Exception(error))
                user_message = ErrorHandler.get_user_friendly_message(
                    Exception(error), error_type
                )
                logger.warning(
                    "Query execution failed",
                    error=error,
                    retries=retries,
                    error_type=error_type.value,
                )

                return QueryResponse(
                    success=False,
                    error=user_message,
                    retries=retries,
                    execution_time=elapsed,
                )

            # Success response
            logger.info(
                "Query executed successfully",
                retries=retries,
                execution_time=elapsed,
            )

            return QueryResponse(
                success=True,
                sql_query=sql_query.query if sql_query else None,
                explanation=sql_query.explanation if sql_query else None,
                execution_result=execution_result,
                retries=retries,
                execution_time=elapsed,
            )

        finally:
            # Restore original MAX_RETRIES
            if original_max_retries:
                os.environ["MAX_RETRIES"] = original_max_retries

    except CircuitBreakerError as e:
        logger.error("Circuit breaker is open", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable due to high error rate. Please try again later.",
        )

    except HTTPException:
        raise

    except Exception as e:
        elapsed = time.time() - start_time
        error_type = ErrorHandler.classify_error(e)
        user_message = ErrorHandler.get_user_friendly_message(e, error_type)

        logger.error(
            "Unexpected error in query endpoint",
            error=str(e),
            error_type=error_type.value,
            execution_time=elapsed,
        )

        metrics.record_query_execution(
            success=False,
            retries=retries,
            response_time=elapsed,
            error_type=error_type.value,
        )

        raise HTTPException(
            status_code=500,
            detail=user_message,
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method,
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected error occurred. Please try again later.",
        },
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
