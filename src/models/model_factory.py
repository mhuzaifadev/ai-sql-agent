"""
Model Factory for LLM initialization with hot-swapping capability.

Supports OpenAI and Google Gemini models with automatic fallback mechanism.
"""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from typing import Optional

# Load environment variables
load_dotenv()


class ModelFactory:
    """Factory class for initializing and managing LLM models."""

    @staticmethod
    def get_llm(
        model_type: Optional[str] = None,
        temperature: float = 0,
        fallback: bool = True,
    ) -> BaseChatModel:
        """
        Initialize LLM with hot-swapping capability.

        Args:
            model_type: Model type ('openai' or 'gemini'). If None, uses DEFAULT_MODEL env var.
            temperature: Temperature setting for the model (default: 0 for deterministic).
            fallback: If True, attempts fallback to alternative model on failure.

        Returns:
            Initialized chat model instance.

        Raises:
            ValueError: If no valid API keys are found or model initialization fails.
        """
        # Determine which model to use
        if model_type is None:
            model_type = os.getenv("DEFAULT_MODEL", "gemini").lower()

        # Try to initialize the requested model
        try:
            if model_type == "openai":
                return ModelFactory._get_openai_llm(temperature)
            elif model_type == "gemini":
                return ModelFactory._get_gemini_llm(temperature)
            else:
                raise ValueError(
                    f"Unknown model type: {model_type}. Supported: 'openai', 'gemini'"
                )
        except Exception as e:
            if not fallback:
                raise

            # Attempt fallback to alternative model
            fallback_model = "gemini" if model_type == "openai" else "openai"
            try:
                print(
                    f"Warning: Failed to initialize {model_type} model: {e}. "
                    f"Attempting fallback to {fallback_model}..."
                )
                if fallback_model == "openai":
                    return ModelFactory._get_openai_llm(temperature)
                else:
                    return ModelFactory._get_gemini_llm(temperature)
            except Exception as fallback_error:
                raise ValueError(
                    f"Failed to initialize both {model_type} and {fallback_model} models. "
                    f"Original error: {e}. Fallback error: {fallback_error}"
                )

    @staticmethod
    def _get_openai_llm(temperature: float) -> ChatOpenAI:
        """
        Initialize OpenAI chat model.

        Args:
            temperature: Temperature setting for the model.

        Returns:
            ChatOpenAI instance.

        Raises:
            ValueError: If OPENAI_API_KEY is not set.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )

        return ChatOpenAI(
            model="gpt-4o-mini",  # Using cost-effective model, can be configured
            temperature=temperature,
            api_key=api_key,
        )

    @staticmethod
    def _get_gemini_llm(temperature: float) -> ChatGoogleGenerativeAI:
        """
        Initialize Google Gemini chat model.

        Args:
            temperature: Temperature setting for the model.

        Returns:
            ChatGoogleGenerativeAI instance.

        Raises:
            ValueError: If GEMINI_API_KEY is not set.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )

        return ChatGoogleGenerativeAI(
            model="gemini-pro",  # Default Gemini model
            temperature=temperature,
            google_api_key=api_key,
        )

    @staticmethod
    def get_available_models() -> list[str]:
        """
        Get list of available models based on configured API keys.

        Returns:
            List of available model names.
        """
        available = []
        if os.getenv("OPENAI_API_KEY"):
            available.append("openai")
        if os.getenv("GEMINI_API_KEY"):
            available.append("gemini")
        return available

