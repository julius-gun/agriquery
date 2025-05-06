from abc import ABC, abstractmethod
from typing import Dict, Any
# REMOVED: from llm_connectors.ollama_connector import OllamaConnector # This caused the circular import

class BaseLLMConnector(ABC):
    """Abstract base class for LLM connectors."""

    @abstractmethod
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the LLM connector.

        Args:
            model_name (str): The name of the LLM model to use.
            config (Dict[str, Any]): Configuration parameters for the LLM.
        """
        self.model_name = model_name
        self.config = config
        self.temperature = config.get("temperature", 0.0) # Default temperature
        self.num_predict = config.get("num_predict", 256)     # Default max tokens
        self.context_window = config.get("context_window")   # Context window size, might be None or specific value

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        """
        Invokes the LLM with the given prompt and returns the response.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The LLM's response.
        """
        pass

    def stop_model(self):
        """
        Optional method to stop or release resources associated with the model.
        Subclasses can override this if specific cleanup is needed.
        By default, it does nothing.
        """
        # print(f"DEBUG: BaseLLMConnector.stop_model() called for {self.model_name}. Default implementation: doing nothing.")
        pass

# Removed the unnecessary import above