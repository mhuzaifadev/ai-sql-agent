"""
SQL syntax validation module.

Validates SQL query syntax and structure using sqlparse.
"""

import sqlparse
from typing import Dict, Any, List
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword


class SQLValidator:
    """Validates SQL query syntax and structure."""

    @staticmethod
    def validate_syntax(sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL query syntax.

        Args:
            sql_query: SQL query string to validate

        Returns:
            Dictionary with validation results:
            - is_valid: bool
            - errors: List[str]
            - warnings: List[str]
            - parsed_statements: List of parsed statements
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not sql_query or not sql_query.strip():
            return {
                "is_valid": False,
                "errors": ["SQL query is empty"],
                "warnings": [],
                "parsed_statements": [],
            }

        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return {
                    "is_valid": False,
                    "errors": ["Failed to parse SQL query"],
                    "warnings": [],
                    "parsed_statements": [],
                }

            # Check for multiple statements (potential security risk)
            if len(parsed) > 1:
                warnings.append(
                    f"Query contains {len(parsed)} statements. Only the first will be executed."
                )

            # Validate first statement
            statement = parsed[0]
            statement_errors = SQLValidator._validate_statement(statement)
            errors.extend(statement_errors)

            return {
                "is_valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "parsed_statements": parsed,
            }

        except Exception as e:
            return {
                "is_valid": False,
                "errors": [f"Syntax validation error: {str(e)}"],
                "warnings": warnings,
                "parsed_statements": [],
            }

    @staticmethod
    def _validate_statement(statement: Statement) -> List[str]:
        """
        Validate a single SQL statement.

        Args:
            statement: Parsed SQL statement

        Returns:
            List of error messages
        """
        errors: List[str] = []

        # Check if it contains SELECT (more lenient than checking first keyword)
        # sqlparse may tokenize differently, so we check the raw string
        statement_str = str(statement).strip().upper()
        
        # Check if it's a SELECT statement (may have whitespace/comments before)
        if not statement_str.startswith("SELECT") and "SELECT" not in statement_str[:20]:
            # Try to find first keyword in tokens
            tokens = list(statement.flatten())
            first_keyword = None
            for token in tokens:
                if token.ttype is Keyword:
                    first_keyword = token.value.upper()
                    break
            
            # Only error if we found a keyword that's not SELECT
            if first_keyword and first_keyword != "SELECT":
                errors.append(
                    f"Query must be a SELECT statement, found: {first_keyword}"
                )

        # Additional validation: Check for basic syntax issues
        # Simple check: if statement contains "SELECT FROM" without columns, it's likely invalid
        statement_lower = statement_str.lower()
        if "select" in statement_lower and "from" in statement_lower:
            # Check if there's content between SELECT and FROM
            select_pos = statement_lower.find("select")
            from_pos = statement_lower.find("from", select_pos)
            if from_pos > select_pos:
                between = statement_lower[select_pos + 6:from_pos].strip()
                # If there's nothing or just whitespace between SELECT and FROM, it's invalid
                # Allow * or column names
                if not between or (not "*" in between and len(between.split()) == 0):
                    errors.append("SELECT statement must specify columns or use *")

        return errors

    @staticmethod
    def format_query(sql_query: str) -> str:
        """
        Format SQL query for readability.

        Args:
            sql_query: SQL query string

        Returns:
            Formatted SQL query string
        """
        try:
            formatted = sqlparse.format(
                sql_query,
                reindent=True,
                keyword_case="upper",
                indent_width=2,
            )
            return formatted
        except Exception:
            return sql_query

