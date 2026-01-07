"""
Pydantic schemas for SQL Compiler Agent.

Defines data structures for SQL queries, validation results, and graph state.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any


class SQLQuery(BaseModel):
    """Schema for SQL query with metadata."""

    query: str = Field(
        description="Syntactically correct PostgreSQL query",
        min_length=1,
    )
    explanation: str = Field(
        description="Brief explanation of what the query does",
        min_length=1,
    )
    risk_score: int = Field(
        ge=1,
        le=10,
        description="Safety score from 1 (safest) to 10 (riskiest)",
        default=5,
    )
    tables_used: List[str] = Field(
        description="List of table names referenced in the query",
        default_factory=list,
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        """Ensure query is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("SQL query cannot be empty")
        return v.strip()

    @field_validator("tables_used")
    @classmethod
    def validate_tables_list(cls, v: List[str]) -> List[str]:
        """Ensure tables list contains valid table names."""
        return [table.strip() for table in v if table.strip()]


class ValidationResult(BaseModel):
    """Schema for SQL validation results."""

    is_valid: bool = Field(description="Whether the SQL query is valid")
    errors: List[str] = Field(
        description="List of validation errors if any",
        default_factory=list,
    )
    warnings: List[str] = Field(
        description="List of validation warnings if any",
        default_factory=list,
    )
    safety_checks_passed: bool = Field(
        description="Whether all safety checks passed",
        default=False,
    )
    risk_assessment: Dict[str, Any] = Field(
        description="Detailed risk assessment",
        default_factory=dict,
    )

    @field_validator("errors", "warnings")
    @classmethod
    def validate_messages(cls, v: List[str]) -> List[str]:
        """Ensure error/warning messages are non-empty strings."""
        return [msg.strip() for msg in v if msg.strip()]


class GraphState(BaseModel):
    """
    Schema for LangGraph state management.

    Note: This is a Pydantic model for validation. In Phase 5, this will be
    converted to a TypedDict for LangGraph compatibility.
    """

    messages: List[Dict[str, Any]] = Field(
        description="List of conversation messages",
        default_factory=list,
    )
    user_question: str = Field(
        description="The original user question in natural language",
        min_length=1,
    )
    relevant_schemas: List[Dict[str, Any]] = Field(
        description="Relevant database table schemas for the query",
        default_factory=list,
    )
    sql_query: Optional[SQLQuery] = Field(
        description="Generated SQL query",
        default=None,
    )
    execution_result: Optional[Dict[str, Any]] = Field(
        description="Query execution results",
        default=None,
    )
    error: Optional[str] = Field(
        description="Error message if any occurred",
        default=None,
    )
    retries: int = Field(
        description="Number of retry attempts",
        ge=0,
        default=0,
    )

    @field_validator("user_question")
    @classmethod
    def validate_user_question(cls, v: str) -> str:
        """Ensure user question is not empty."""
        if not v or not v.strip():
            raise ValueError("User question cannot be empty")
        return v.strip()

    @field_validator("retries")
    @classmethod
    def validate_retries(cls, v: int) -> int:
        """Ensure retries is non-negative."""
        if v < 0:
            raise ValueError("Retries must be non-negative")
        return v

