"""
Security validation module for SQL queries.

Implements keyword blacklist checking, query complexity analysis,
table access validation, and SQL injection pattern detection.
"""

import re
from typing import List, Dict, Any, Set, Tuple, Optional
import sqlparse
from sqlparse.sql import Statement, Token
from sqlparse.tokens import Keyword, Name


# Forbidden SQL keywords that should never appear in queries
FORBIDDEN_KEYWORDS = [
    "DROP",
    "DELETE",
    "TRUNCATE",
    "ALTER",
    "CREATE",
    "GRANT",
    "REVOKE",
    "INSERT",
    "UPDATE",
    "EXEC",
    "EXECUTE",
    "CALL",
    "COPY",
    "TRANSACTION",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "LOCK",
    "UNLOCK",
]

# SQL injection patterns to detect
SQL_INJECTION_PATTERNS = [
    r"(--|#|/\*|\*/)",  # SQL comments
    r"(\bor\b|\band\b)\s*\d+\s*=\s*\d+",  # Boolean-based injection
    r"\bunion\s+(all\s+)?select\b",  # UNION-based injection (more specific)
    r"';?\s*(drop|delete|insert|update)",  # Query termination attempts
    r"(\bor\b|\band\b)\s*['\"]?\s*\d+\s*=\s*['\"]?\s*\d+",  # Boolean logic injection
    r"exec\s*\(",  # Function execution attempts
    r"xp_\w+",  # Extended stored procedures
    r"sp_\w+",  # Stored procedures
    r"waitfor\s+delay",  # Time-based injection
    r"benchmark\s*\(",  # MySQL time-based injection
]


class SafetyChecker:
    """Validates SQL queries for security and safety."""

    def __init__(self, allowed_tables: Optional[Set[str]] = None):
        """
        Initialize safety checker.

        Args:
            allowed_tables: Set of allowed table names. If None, all tables are allowed.
        """
        self.allowed_tables = allowed_tables
        self.forbidden_keywords_lower = {kw.lower() for kw in FORBIDDEN_KEYWORDS}

    def check_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Perform comprehensive safety checks on a SQL query.

        Args:
            sql_query: SQL query string to validate

        Returns:
            Dictionary with validation results:
            - is_safe: bool
            - errors: List[str]
            - warnings: List[str]
            - risk_score: int (1-10)
            - details: Dict with specific check results
        """
        errors: List[str] = []
        warnings: List[str] = []
        risk_score = 1  # Start with lowest risk

        # Normalize query
        normalized_query = sql_query.strip()

        # Check 1: Forbidden keywords
        keyword_result = self._check_forbidden_keywords(normalized_query)
        if not keyword_result["is_safe"]:
            errors.extend(keyword_result["errors"])
            risk_score = max(risk_score, 10)  # Critical risk

        # Check 2: SQL injection patterns
        injection_result = self._check_sql_injection(normalized_query)
        if not injection_result["is_safe"]:
            errors.extend(injection_result["errors"])
            risk_score = max(risk_score, 10)  # Critical risk

        # Check 3: Query complexity
        complexity_result = self._analyze_complexity(normalized_query)
        if complexity_result["risk_score"] > 7:
            warnings.append(
                f"Query complexity is high: {complexity_result['reason']}"
            )
            risk_score = max(risk_score, complexity_result["risk_score"])

        # Check 4: Table access validation
        if self.allowed_tables:
            table_result = self._validate_table_access(normalized_query)
            if not table_result["is_safe"]:
                errors.extend(table_result["errors"])
                risk_score = max(risk_score, 8)

        # Check 5: Ensure it's a SELECT query
        if not normalized_query.upper().strip().startswith("SELECT"):
            errors.append("Only SELECT queries are allowed")
            risk_score = max(risk_score, 10)

        return {
            "is_safe": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "risk_score": min(risk_score, 10),  # Cap at 10
            "details": {
                "keyword_check": keyword_result,
                "injection_check": injection_result,
                "complexity_check": complexity_result,
                "table_check": table_result if self.allowed_tables else None,
            },
        }

    def _check_forbidden_keywords(self, sql_query: str) -> Dict[str, Any]:
        """
        Check for forbidden SQL keywords.

        Args:
            sql_query: SQL query to check

        Returns:
            Dictionary with check results
        """
        query_upper = sql_query.upper()
        found_keywords = []

        for keyword in FORBIDDEN_KEYWORDS:
            # Use word boundaries to avoid false positives
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, query_upper, re.IGNORECASE):
                found_keywords.append(keyword)

        is_safe = len(found_keywords) == 0
        errors = []
        if not is_safe:
            errors.append(
                f"Forbidden keywords detected: {', '.join(found_keywords)}"
            )

        return {
            "is_safe": is_safe,
            "errors": errors,
            "found_keywords": found_keywords,
        }

    def _check_sql_injection(self, sql_query: str) -> Dict[str, Any]:
        """
        Check for SQL injection patterns.

        Args:
            sql_query: SQL query to check

        Returns:
            Dictionary with check results
        """
        found_patterns = []
        query_lower = sql_query.lower()

        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                found_patterns.append(pattern)

        is_safe = len(found_patterns) == 0
        errors = []
        if not is_safe:
            errors.append(
                f"Potential SQL injection patterns detected: {len(found_patterns)} pattern(s)"
            )

        return {
            "is_safe": is_safe,
            "errors": errors,
            "found_patterns": found_patterns,
        }

    def _analyze_complexity(self, sql_query: str) -> Dict[str, Any]:
        """
        Analyze query complexity.

        Args:
            sql_query: SQL query to analyze

        Returns:
            Dictionary with complexity analysis and risk score
        """
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return {
                    "risk_score": 5,
                    "reason": "Could not parse query",
                    "details": {},
                }

            statement = parsed[0]
            complexity_score = 0
            details = {}

            # Count JOINs - look for JOIN keyword (INNER JOIN, LEFT JOIN, etc. all have JOIN)
            join_count = 0
            tokens_list = list(statement.flatten())
            skip_next = False
            for i, token in enumerate(tokens_list):
                if skip_next:
                    skip_next = False
                    continue
                    
                if token.ttype is Keyword:
                    token_upper = token.value.upper()
                    # Count actual JOIN keywords
                    if token_upper == "JOIN":
                        join_count += 1
                    # Also count if we see INNER/LEFT/etc followed by JOIN (but don't double count)
                    elif token_upper in ("INNER", "LEFT", "RIGHT", "FULL", "OUTER"):
                        # Check if next token is JOIN
                        if i + 1 < len(tokens_list):
                            next_token = tokens_list[i + 1]
                            if next_token.ttype is Keyword and next_token.value.upper() == "JOIN":
                                join_count += 1
                                skip_next = True  # Skip the JOIN token to avoid double counting
            
            details["joins"] = join_count
            complexity_score += join_count * 1

            # Count subqueries - look for SELECT keywords
            select_count = len(
                [
                    token
                    for token in statement.flatten()
                    if token.ttype is Keyword and token.value.upper() == "SELECT"
                ]
            )
            # Subtract 1 for the main SELECT
            subquery_count = max(0, select_count - 1)
            details["subqueries"] = subquery_count
            complexity_score += subquery_count * 2

            # Check for UNION
            has_union = any(
                token.ttype is Keyword and token.value.upper() in ("UNION", "UNION ALL")
                for token in statement.flatten()
            )
            details["has_union"] = has_union
            if has_union:
                complexity_score += 2

            # Check query length
            query_length = len(sql_query)
            details["query_length"] = query_length
            if query_length > 5000:
                complexity_score += 3
            elif query_length > 2000:
                complexity_score += 1

            # Calculate risk score (1-10 scale)
            risk_score = min(1 + complexity_score, 10)

            reason = ""
            if risk_score >= 8:
                reason = "Very high complexity (multiple joins/subqueries or very long query)"
            elif risk_score >= 5:
                reason = "Moderate complexity"
            else:
                reason = "Low complexity"

            return {
                "risk_score": risk_score,
                "reason": reason,
                "details": details,
            }

        except Exception as e:
            return {
                "risk_score": 5,
                "reason": f"Error analyzing complexity: {e}",
                "details": {},
            }

    def _validate_table_access(self, sql_query: str) -> Dict[str, Any]:
        """
        Validate that query only accesses allowed tables.

        Args:
            sql_query: SQL query to validate

        Returns:
            Dictionary with validation results
        """
        try:
            parsed = sqlparse.parse(sql_query)
            if not parsed:
                return {
                    "is_safe": False,
                    "errors": ["Could not parse query to extract tables"],
                }

            statement = parsed[0]
            accessed_tables = self._extract_table_names(statement)

            unauthorized_tables = [
                table
                for table in accessed_tables
                if table not in self.allowed_tables
            ]

            is_safe = len(unauthorized_tables) == 0
            errors = []
            if not is_safe:
                errors.append(
                    f"Unauthorized table access: {', '.join(unauthorized_tables)}"
                )

            return {
                "is_safe": is_safe,
                "errors": errors,
                "accessed_tables": list(accessed_tables),
                "unauthorized_tables": unauthorized_tables,
            }

        except Exception as e:
            return {
                "is_safe": False,
                "errors": [f"Error validating table access: {e}"],
            }

    def _extract_table_names(self, statement: Statement) -> Set[str]:
        """
        Extract table names from a parsed SQL statement.

        Args:
            statement: Parsed SQL statement

        Returns:
            Set of table names
        """
        tables = set()
        from_seen = False

        for token in statement.flatten():
            if token.ttype is Keyword and token.value.upper() == "FROM":
                from_seen = True
                continue

            if from_seen and token.ttype is Name:
                # Extract table name (handle schema.table format)
                table_name = token.value.split(".")[-1].strip("`\"'[]")
                if table_name:
                    tables.add(table_name.lower())

            # Reset on next keyword (JOIN, WHERE, etc.)
            if (
                from_seen
                and token.ttype is Keyword
                and token.value.upper()
                in ("WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "JOIN")
            ):
                from_seen = False

        return tables

