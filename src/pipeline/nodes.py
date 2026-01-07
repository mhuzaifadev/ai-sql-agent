"""
Graph nodes for the SQL Compiler Agent pipeline.

Each node performs a specific step in the SQL generation and execution process.
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser

from src.pipeline.state import AgentState, log_state
from src.pipeline.schema_router import SchemaRouter, ContextBuilder
from src.pipeline.prompts import (
    SQL_GENERATION_SYSTEM_PROMPT,
    SQL_GENERATION_USER_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    REFLECTION_USER_PROMPT,
)
from src.models.model_factory import ModelFactory
from src.models.schemas import SQLQuery, ValidationResult
from src.validators.safety_checks import SafetyChecker
from src.validators.sql_validator import SQLValidator
from src.database.connection import DatabaseConnection
from src.utils.logger import get_logger, get_metrics, track_execution_time
from src.utils.error_handler import (
    get_llm_circuit_breaker,
    CircuitBreakerError,
    ErrorHandler,
    ErrorType,
)

logger = get_logger()
metrics = get_metrics()


# Global instances (will be initialized in graph setup)
schema_router: SchemaRouter = None
context_builder: ContextBuilder = None
db_connection: DatabaseConnection = None
safety_checker: SafetyChecker = None
sql_validator: SQLValidator = None
llm = None


def initialize_nodes(
    db_conn: DatabaseConnection,
    schema_router_instance: SchemaRouter,
    context_builder_instance: ContextBuilder,
) -> None:
    """
    Initialize global node dependencies.

    Args:
        db_conn: Database connection instance
        schema_router_instance: Schema router instance
        context_builder_instance: Context builder instance
    """
    global schema_router, context_builder, db_connection, safety_checker, sql_validator, llm

    schema_router = schema_router_instance
    context_builder = context_builder_instance
    db_connection = db_conn
    safety_checker = SafetyChecker()
    sql_validator = SQLValidator()
    llm = ModelFactory.get_llm()


def route_schema_node(state: AgentState) -> AgentState:
    """
    Route user question to relevant database schemas using RAG.

    Args:
        state: Current agent state

    Returns:
        Updated state with relevant_schemas populated
    """
    log_state(state, "ROUTE_SCHEMA")

    user_question = state["user_question"]

    try:
        # Find relevant schemas using similarity search
        relevant_schemas = schema_router.find_relevant_schemas(user_question)

        # Update state
        state["relevant_schemas"] = relevant_schemas
        state["messages"].append(
            HumanMessage(content=f"Found {len(relevant_schemas)} relevant tables")
        )

        return state
    except Exception as e:
        state["error"] = f"Schema routing failed: {str(e)}"
        return state


def generate_sql_node(state: AgentState) -> AgentState:
    """
    Generate SQL query from user question and relevant schemas.

    Args:
        state: Current agent state

    Returns:
        Updated state with sql_query populated
    """
    log_state(state, "GENERATE_SQL")
    logger.info("Generating SQL query", question=state["user_question"][:100])

    user_question = state["user_question"]
    relevant_schemas = state["relevant_schemas"]

    try:
        # Use circuit breaker for LLM calls
        circuit_breaker = get_llm_circuit_breaker()
        
        with track_execution_time("sql_generation"):
            # Build schema context
            schema_context = context_builder.build_schema_context(relevant_schemas)

            # Create prompt
            system_prompt = SQL_GENERATION_SYSTEM_PROMPT
            user_prompt = SQL_GENERATION_USER_PROMPT.format(
                schemas=schema_context, question=user_question
            )

            # Use structured output parser
            parser = PydanticOutputParser(pydantic_object=SQLQuery)
            format_instructions = parser.get_format_instructions()

            # Add format instructions to prompt
            enhanced_prompt = user_prompt + f"\n\n{format_instructions}"

            # Generate response using LangChain message format
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=enhanced_prompt),
            ]

            def _call_llm():
                return llm.invoke(messages)

            response = circuit_breaker.call(_call_llm)
            
            # Track LLM usage (approximate token count)
            response_text = response.content if hasattr(response, "content") else str(response)
            estimated_tokens = len(response_text.split()) * 1.3  # Rough estimate
            metrics.record_llm_call(tokens_used=int(estimated_tokens))

            # Try to extract SQL query from response
            # The LLM might return JSON or just SQL
            try:
                sql_query = parser.parse(response_text)
            except Exception:
                # Fallback: try to extract SQL query directly
                # Look for SQL between ```sql and ``` or just assume the whole response is SQL
                if "```sql" in response_text:
                    sql_text = response_text.split("```sql")[1].split("```")[0].strip()
                elif "```" in response_text:
                    sql_text = response_text.split("```")[1].split("```")[0].strip()
                else:
                    sql_text = response_text.strip()

                # Create SQLQuery object manually
                sql_query = SQLQuery(
                    query=sql_text,
                    explanation="Generated SQL query",
                    risk_score=5,
                    tables_used=[schema.get("table_name", "") for schema in relevant_schemas],
                )

            # Update state
            state["sql_query"] = sql_query
            state["messages"].append(
                AIMessage(content=f"Generated SQL query: {sql_query.query[:100]}...")
            )
            
            logger.info("SQL query generated successfully", query_preview=sql_query.query[:50])

            return state
    except CircuitBreakerError as e:
        logger.error("Circuit breaker is open, cannot generate SQL", error=str(e))
        state["error"] = "AI service temporarily unavailable. Please try again later."
        return state
    except Exception as e:
        error_type = ErrorHandler.classify_error(e)
        logger.error("SQL generation failed", error=str(e), error_type=error_type.value)
        state["error"] = f"SQL generation failed: {str(e)}"
        return state


def validate_sql_node(state: AgentState) -> AgentState:
    """
    Validate generated SQL query for safety and syntax.

    Args:
        state: Current agent state

    Returns:
        Updated state with validation result
    """
    log_state(state, "VALIDATE_SQL")

    sql_query = state.get("sql_query")
    if not sql_query:
        state["error"] = "No SQL query to validate"
        return state

    try:
        # Syntax validation
        syntax_result = sql_validator.validate_syntax(sql_query.query)

        # Safety checks
        safety_result = safety_checker.check_query(sql_query.query)

        # Combine results
        is_valid = syntax_result["is_valid"] and safety_result["is_safe"]
        errors = syntax_result["errors"] + safety_result["errors"]
        warnings = syntax_result["warnings"] + safety_result["warnings"]

        validation_result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            safety_checks_passed=safety_result["is_safe"],
            risk_assessment=safety_result,
        )

        # Update state
        if not is_valid:
            state["error"] = "; ".join(errors)
        else:
            state["error"] = None

        state["messages"].append(
            AIMessage(
                content=f"Validation: {'PASSED' if is_valid else 'FAILED'} - {len(warnings)} warnings"
            )
        )

        return state
    except Exception as e:
        state["error"] = f"Validation failed: {str(e)}"
        return state


def execute_sql_node(state: AgentState) -> AgentState:
    """
    Execute validated SQL query against the database.

    Args:
        state: Current agent state

    Returns:
        Updated state with execution_result populated
    """
    log_state(state, "EXECUTE_SQL")

    sql_query = state.get("sql_query")
    if not sql_query:
        state["error"] = "No SQL query to execute"
        return state

    try:
        # Execute query
        results = db_connection.execute_query(sql_query.query)

        # Update state
        state["execution_result"] = {
            "success": True,
            "row_count": len(results),
            "data": results[:100],  # Limit to first 100 rows
            "truncated": len(results) > 100,
        }

        state["error"] = None
        state["messages"].append(
            AIMessage(content=f"Query executed successfully: {len(results)} rows returned")
        )

        return state
    except Exception as e:
        state["error"] = f"Query execution failed: {str(e)}"
        state["execution_result"] = {
            "success": False,
            "error": str(e),
        }
        return state


def reflect_node(state: AgentState) -> AgentState:
    """
    Reflect on error and generate corrected SQL query.

    Args:
        state: Current agent state

    Returns:
        Updated state with corrected sql_query and incremented retries
    """
    log_state(state, "REFLECT")
    logger.info("Reflecting on error", retries=state.get("retries", 0), error=state.get("error", "Unknown")[:100])

    user_question = state["user_question"]
    error = state.get("error", "Unknown error")
    previous_query = state.get("sql_query")
    relevant_schemas = state.get("relevant_schemas", [])
    retries = state.get("retries", 0)

    try:
        # Use circuit breaker for LLM calls
        circuit_breaker = get_llm_circuit_breaker()
        
        with track_execution_time("reflection"):
            # Increment retries
            state["retries"] = retries + 1

            # Build schema context
            schema_context = context_builder.build_schema_context(relevant_schemas)

            # Create reflection prompt
            system_prompt = REFLECTION_SYSTEM_PROMPT
            user_prompt = REFLECTION_USER_PROMPT.format(
                error=error,
                previous_query=previous_query.query if previous_query else "N/A",
                schemas=schema_context,
                question=user_question,
            )

            # Generate corrected SQL using LangChain message format
            from langchain_core.messages import SystemMessage, HumanMessage

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            def _call_llm():
                return llm.invoke(messages)

            response = circuit_breaker.call(_call_llm)
            
            # Track LLM usage
            response_text = response.content if hasattr(response, "content") else str(response)
            estimated_tokens = len(response_text.split()) * 1.3
            metrics.record_llm_call(tokens_used=int(estimated_tokens))

            # Try to extract SQL query from response
            if "```sql" in response_text:
                sql_text = response_text.split("```sql")[1].split("```")[0].strip()
            elif "```" in response_text:
                sql_text = response_text.split("```")[1].split("```")[0].strip()
            else:
                # Try to find SQL-like text
                lines = response_text.split("\n")
                sql_text = "\n".join(
                    [line for line in lines if any(keyword in line.upper() for keyword in ["SELECT", "FROM", "WHERE", "JOIN"])]
                ).strip()
                if not sql_text:
                    sql_text = response_text.strip()

            # Create corrected SQLQuery
            corrected_query = SQLQuery(
                query=sql_text,
                explanation=f"Corrected query after reflection (attempt {state['retries']})",
                risk_score=previous_query.risk_score if previous_query else 5,
                tables_used=[schema.get("table_name", "") for schema in relevant_schemas],
            )

            # Update state
            state["sql_query"] = corrected_query
            state["error"] = None  # Clear error for retry
            state["messages"].append(
                AIMessage(content=f"Generated corrected query (retry {state['retries']})")
            )
            
            logger.info("Reflection completed", retries=state["retries"])

            return state
    except CircuitBreakerError as e:
        logger.error("Circuit breaker is open during reflection", error=str(e))
        state["error"] = "AI service temporarily unavailable. Cannot retry."
        return state
    except Exception as e:
        error_type = ErrorHandler.classify_error(e)
        logger.error("Reflection failed", error=str(e), error_type=error_type.value)
        state["error"] = f"Reflection failed: {str(e)}"
        return state

