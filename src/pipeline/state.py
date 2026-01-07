"""
Graph state definition for LangGraph pipeline.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph.message import add_messages

from src.models.schemas import SQLQuery


class AgentState(TypedDict):
    """
    State structure for the SQL Compiler Agent graph.

    This TypedDict defines the state that flows through the LangGraph pipeline.
    """

    messages: Annotated[List[Dict[str, Any]], add_messages]
    user_question: str
    relevant_schemas: List[Dict[str, Any]]
    sql_query: Optional[SQLQuery]
    execution_result: Optional[Dict[str, Any]]
    error: Optional[str]
    retries: int


def create_initial_state(user_question: str) -> AgentState:
    """
    Create initial state for the agent graph.

    Args:
        user_question: The user's natural language question

    Returns:
        Initial AgentState dictionary
    """
    return {
        "messages": [],
        "user_question": user_question,
        "relevant_schemas": [],
        "sql_query": None,
        "execution_result": None,
        "error": None,
        "retries": 0,
    }


def log_state(state: AgentState, step_name: str) -> None:
    """
    Log state for debugging purposes.

    Args:
        state: Current agent state
        step_name: Name of the current step
    """
    print(f"\n[{step_name}] State:")
    print(f"  User Question: {state.get('user_question', 'N/A')}")
    print(f"  Relevant Schemas: {len(state.get('relevant_schemas', []))} tables")
    print(f"  SQL Query: {state.get('sql_query') is not None}")
    print(f"  Execution Result: {state.get('execution_result') is not None}")
    print(f"  Error: {state.get('error', 'None')}")
    print(f"  Retries: {state.get('retries', 0)}")

