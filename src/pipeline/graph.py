"""
LangGraph pipeline for SQL Compiler Agent.

Defines the complete graph structure with nodes and conditional edges.
"""

from typing import Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.pipeline.state import AgentState
from src.pipeline.nodes import (
    route_schema_node,
    generate_sql_node,
    validate_sql_node,
    execute_sql_node,
    reflect_node,
    initialize_nodes,
)
from src.database.connection import DatabaseConnection
from src.pipeline.schema_router import SchemaRouter, ContextBuilder
import os


def should_retry(state: AgentState) -> Literal["reflect", "end"]:
    """
    Conditional edge function: decide whether to retry or end.

    Args:
        state: Current agent state

    Returns:
        "reflect" if should retry, "end" if max retries reached
    """
    max_retries = int(os.getenv("MAX_RETRIES", "3"))
    retries = state.get("retries", 0)
    error = state.get("error")

    if error and retries < max_retries:
        return "reflect"
    else:
        return "end"


def should_continue(state: AgentState) -> Literal["execute_sql", "reflect", "end"]:
    """
    Conditional edge function: decide next step after validation.

    Args:
        state: Current agent state

    Returns:
        "execute_sql" if valid, "reflect" if invalid and can retry, "end" if max retries
    """
    error = state.get("error")
    retries = state.get("retries", 0)
    max_retries = int(os.getenv("MAX_RETRIES", "3"))

    if not error:
        # Validation passed, execute query
        return "execute_sql"
    elif retries < max_retries:
        # Validation failed but can retry
        return "reflect"
    else:
        # Max retries reached
        return "end"


class SQLCompilerGraph:
    """Main graph class for SQL Compiler Agent."""

    def __init__(
        self,
        db_connection: DatabaseConnection,
        schema_router: Optional[SchemaRouter] = None,
        context_builder: Optional[ContextBuilder] = None,
    ):
        """
        Initialize the SQL Compiler graph.

        Args:
            db_connection: Database connection instance
            schema_router: Optional schema router (will create if not provided)
            context_builder: Optional context builder (will create if not provided)
        """
        self.db_connection = db_connection

        # Initialize schema router and context builder if not provided
        if schema_router is None:
            schema_router = SchemaRouter(db_connection)
        if context_builder is None:
            context_builder = ContextBuilder()

        self.schema_router = schema_router
        self.context_builder = context_builder

        # Initialize node dependencies
        initialize_nodes(db_connection, schema_router, context_builder)

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the complete LangGraph structure.

        Returns:
            Compiled StateGraph
        """
        # Create graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("route_schema", route_schema_node)
        workflow.add_node("generate_sql", generate_sql_node)
        workflow.add_node("validate_sql", validate_sql_node)
        workflow.add_node("execute_sql", execute_sql_node)
        workflow.add_node("reflect", reflect_node)

        # Define edges
        # Start -> route_schema
        workflow.set_entry_point("route_schema")

        # route_schema -> generate_sql
        workflow.add_edge("route_schema", "generate_sql")

        # generate_sql -> validate_sql
        workflow.add_edge("generate_sql", "validate_sql")

        # validate_sql -> conditional (execute_sql, reflect, or end)
        workflow.add_conditional_edges(
            "validate_sql",
            should_continue,
            {
                "execute_sql": "execute_sql",
                "reflect": "reflect",
                "end": END,
            },
        )

        # execute_sql -> end (success)
        workflow.add_edge("execute_sql", END)

        # reflect -> conditional (generate_sql or end)
        workflow.add_conditional_edges(
            "reflect",
            should_retry,
            {
                "reflect": "generate_sql",  # Retry by generating new SQL
                "end": END,
            },
        )

        # Compile graph with memory
        memory = MemorySaver()
        compiled_graph = workflow.compile(checkpointer=memory)

        return compiled_graph

    def invoke(self, user_question: str, config: Optional[dict] = None) -> AgentState:
        """
        Invoke the graph with a user question.

        Args:
            user_question: Natural language question
            config: Optional LangGraph configuration

        Returns:
            Final agent state
        """
        from src.pipeline.state import create_initial_state

        initial_state = create_initial_state(user_question)

        if config is None:
            config = {"configurable": {"thread_id": "default"}}

        # Run the graph
        final_state = self.graph.invoke(initial_state, config)

        return final_state

    def stream(self, user_question: str, config: Optional[dict] = None):
        """
        Stream the graph execution for real-time updates.

        Args:
            user_question: Natural language question
            config: Optional LangGraph configuration

        Yields:
            State updates as the graph executes
        """
        from src.pipeline.state import create_initial_state

        initial_state = create_initial_state(user_question)

        if config is None:
            config = {"configurable": {"thread_id": "default"}}

        # Stream execution
        for event in self.graph.stream(initial_state, config):
            yield event


def create_graph(
    db_connection: Optional[DatabaseConnection] = None,
) -> SQLCompilerGraph:
    """
    Factory function to create a SQL Compiler graph.

    Args:
        db_connection: Optional database connection (will create if not provided)

    Returns:
        Initialized SQLCompilerGraph instance
    """
    if db_connection is None:
        db_connection = DatabaseConnection()
        db_connection.connect()

    return SQLCompilerGraph(db_connection)

