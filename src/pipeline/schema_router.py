"""
Schema Router with RAG (Retrieval-Augmented Generation).

Extracts database schemas, creates embeddings, and performs similarity search
to find relevant tables for a given natural language question.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
from dataclasses import dataclass
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from src.database.connection import DatabaseConnection
from src.models.model_factory import ModelFactory


@dataclass
class SchemaInfo:
    """Information about a database table schema."""

    table_name: str
    columns: List[Dict[str, Any]]
    description: str  # Generated description for embedding


class SchemaRouter:
    """Routes user questions to relevant database schemas using RAG."""

    def __init__(
        self,
        db_connection: DatabaseConnection,
        embedding_model: Optional[str] = None,
        top_k: int = 5,
    ):
        """
        Initialize schema router.

        Args:
            db_connection: Database connection instance
            embedding_model: Model type for embeddings ('openai' or 'gemini')
            top_k: Number of top relevant schemas to return
        """
        self.db_connection = db_connection
        self.top_k = top_k
        self.embeddings: Optional[Embeddings] = None
        self.vector_store: Optional[InMemoryVectorStore] = None
        self.schemas: List[SchemaInfo] = []
        self._initialize_embeddings(embedding_model)

    def _initialize_embeddings(self, model_type: Optional[str] = None) -> None:
        """Initialize embedding model."""
        if model_type is None:
            model_type = os.getenv("DEFAULT_MODEL", "gemini").lower()

        try:
            if model_type == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.embeddings = OpenAIEmbeddings(
                        model="text-embedding-3-small", api_key=api_key
                    )
                else:
                    raise ValueError("OPENAI_API_KEY not found")
            elif model_type == "gemini":
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    self.embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001", google_api_key=api_key
                    )
                else:
                    raise ValueError("GEMINI_API_KEY not found")
            else:
                raise ValueError(f"Unknown embedding model: {model_type}")
        except Exception as e:
            # Fallback to alternative model
            fallback = "gemini" if model_type == "openai" else "openai"
            print(f"Warning: Failed to initialize {model_type} embeddings: {e}")
            print(f"Attempting fallback to {fallback}...")
            self._initialize_embeddings(fallback)

    def extract_schemas(self, schema_name: str = "public") -> List[SchemaInfo]:
        """
        Extract all table schemas from the database.

        Args:
            schema_name: Database schema name (default: 'public')

        Returns:
            List of SchemaInfo objects
        """
        try:
            raw_schemas = self.db_connection.get_table_schemas(schema_name)
            self.schemas = []

            for schema in raw_schemas:
                table_name = schema["table_name"]
                columns = schema.get("columns", [])

                # Generate description for embedding
                description = self._generate_schema_description(table_name, columns)

                schema_info = SchemaInfo(
                    table_name=table_name,
                    columns=columns,
                    description=description,
                )
                self.schemas.append(schema_info)

            return self.schemas
        except Exception as e:
            raise RuntimeError(f"Failed to extract schemas: {e}")

    def _generate_schema_description(
        self, table_name: str, columns: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a natural language description of a table schema for embedding.

        Args:
            table_name: Name of the table
            columns: List of column dictionaries

        Returns:
            Natural language description
        """
        column_descriptions = []
        for col in columns:
            col_name = col.get("column_name", "")
            data_type = col.get("data_type", "")
            is_nullable = col.get("is_nullable", "NO")
            nullable_text = "nullable" if is_nullable == "YES" else "not null"
            column_descriptions.append(f"{col_name} ({data_type}, {nullable_text})")

        description = f"Table {table_name} with columns: {', '.join(column_descriptions)}"
        return description

    def build_vector_store(self) -> None:
        """Build in-memory vector store from extracted schemas."""
        if not self.schemas:
            self.extract_schemas()

        if not self.embeddings:
            raise ValueError("Embeddings not initialized")

        # Create documents from schemas
        documents = []
        for schema in self.schemas:
            # Create a more detailed document for better retrieval
            doc_text = f"{schema.description}\nTable: {schema.table_name}\n"
            doc_text += f"Columns: {', '.join([col.get('column_name', '') for col in schema.columns])}"

            doc = Document(
                page_content=doc_text,
                metadata={
                    "table_name": schema.table_name,
                    "schema_info": schema,
                },
            )
            documents.append(doc)

        # Create vector store
        self.vector_store = InMemoryVectorStore.from_documents(
            documents, self.embeddings
        )

    def find_relevant_schemas(
        self, question: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find relevant schemas for a given question using similarity search.

        Args:
            question: Natural language question
            top_k: Number of top results (defaults to instance top_k)

        Returns:
            List of relevant schema dictionaries
        """
        if not self.vector_store:
            self.build_vector_store()

        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        k = top_k or self.top_k

        # Perform similarity search
        results = self.vector_store.similarity_search(question, k=k)

        # Extract schema information
        relevant_schemas = []
        for result in results:
            schema_info = result.metadata.get("schema_info")
            if schema_info:
                relevant_schemas.append(
                    {
                        "table_name": schema_info.table_name,
                        "columns": schema_info.columns,
                        "description": schema_info.description,
                    }
                )

        return relevant_schemas

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all extracted schemas.

        Returns:
            List of all schema dictionaries
        """
        if not self.schemas:
            self.extract_schemas()

        return [
            {
                "table_name": schema.table_name,
                "columns": schema.columns,
                "description": schema.description,
            }
            for schema in self.schemas
        ]


class ContextBuilder:
    """Builds context strings from schemas with token counting and summarization."""

    def __init__(self, max_tokens: int = 2000):
        """
        Initialize context builder.

        Args:
            max_tokens: Maximum number of tokens for context (approximate)
        """
        self.max_tokens = max_tokens
        # Rough estimate: 1 token ≈ 4 characters
        self.max_chars = max_tokens * 4

    def build_schema_context(
        self, schemas: List[Dict[str, Any]], summarize: bool = True
    ) -> str:
        """
        Build schema context string from relevant schemas.

        Args:
            schemas: List of schema dictionaries
            summarize: Whether to summarize large schemas

        Returns:
            Formatted schema context string
        """
        context_parts = []

        for schema in schemas:
            table_name = schema.get("table_name", "unknown")
            columns = schema.get("columns", [])
            description = schema.get("description", "")

            # Build table description
            table_info = f"Table: {table_name}\n"
            table_info += f"Description: {description}\n"
            table_info += "Columns:\n"

            # Add column details
            for col in columns:
                col_name = col.get("column_name", "")
                data_type = col.get("data_type", "")
                is_nullable = col.get("is_nullable", "NO")
                default_val = col.get("column_default")

                col_info = f"  - {col_name}: {data_type}"
                if is_nullable == "YES":
                    col_info += " (nullable)"
                if default_val:
                    col_info += f" (default: {default_val})"
                table_info += col_info + "\n"

            context_parts.append(table_info)

        context = "\n\n".join(context_parts)

        # Check if we need to summarize
        if summarize and len(context) > self.max_chars:
            context = self._summarize_context(context, schemas)

        return context

    def _summarize_context(
        self, full_context: str, schemas: List[Dict[str, Any]]
    ) -> str:
        """
        Summarize context when it exceeds token limits.

        Args:
            full_context: Full context string
            schemas: List of schemas

        Returns:
            Summarized context string
        """
        # Prioritize: include all table names and key columns
        summary_parts = []
        summary_parts.append(
            f"Summary: {len(schemas)} relevant tables found. Details below:\n"
        )

        for schema in schemas:
            table_name = schema.get("table_name", "unknown")
            columns = schema.get("columns", [])
            # Include only first 10 columns to save space
            key_columns = columns[:10]
            col_names = [col.get("column_name", "") for col in key_columns]

            table_summary = f"Table: {table_name}\n"
            table_summary += f"Key columns: {', '.join(col_names)}"
            if len(columns) > 10:
                table_summary += f" (and {len(columns) - 10} more columns)"
            summary_parts.append(table_summary)

        return "\n\n".join(summary_parts)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

