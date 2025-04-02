# utils/llm_connector_manager.py
from llm_connectors.ollama_connector import OllamaConnector
from llm_connectors.gemini_connector import GeminiConnector
from typing import Dict


class LLMConnectorManager:
    """Manages LLM connector creation based on configuration."""

    def __init__(self, llm_configs: Dict):
        """
        Initializes the LLMConnectorManager with configurations for different LLMs.
        llm_configs should be a dictionary where keys are llm types ('ollama', 'gemini')
        and values are their respective model configurations.
        """
        self.llm_configs = llm_configs

    def get_connector(self, llm_type: str, model_name: str):
        """
        Returns the appropriate LLM connector based on the specified type and model name.

        Args:
            llm_type (str): The type of LLM ('ollama' or 'gemini').
            model_name (str): The name of the LLM model.

        Returns:
            An LLMConnector instance (OllamaConnector or GeminiConnector).

        Raises:
            ValueError: If the LLM type is unsupported or the model is not configured.
        """
        llm_type = llm_type.lower()
        if llm_type not in self.llm_configs:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

        model_config = self.llm_configs[llm_type].get(model_name)
        if not model_config:
            raise ValueError(
                f"Model '{model_name}' not found in configuration for LLM type '{llm_type}'."
            )
        if "name" not in model_config:
            raise ValueError(
                f"Model '{model_name}' not properly configured for LLM type '{llm_type}'. Must have a 'name' key."
            )

        if llm_type == "ollama":
            actual_model_name = model_config["name"]
            # return OllamaConnector(actual_model_name, model_config)
            # Initialize and store OllamaConnector instance for reuse
            connector_instance = OllamaConnector(actual_model_name, model_config)
            return connector_instance
        elif llm_type == "gemini":
            return GeminiConnector(model_name, model_config)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
