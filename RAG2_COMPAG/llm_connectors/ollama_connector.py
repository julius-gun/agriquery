import time
from typing import Dict, Any
from langchain_ollama.llms import OllamaLLM
from llm_connectors.base_llm_connector import BaseLLMConnector
from ollama._types import ResponseError


class OllamaConnector(BaseLLMConnector):
    """Connector for Ollama LLMs."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the OllamaConnector.

        Args:
            model_name (str): The *key* of the model in the config (can be a user-defined alias).
            config (Dict[str, Any]): Configuration parameters for the Ollama model.
                                    Should include 'name' (the *actual* Ollama model name),
                                    and optionally 'temperature', 'num_predict', 'context_window',
                                    and 'num_gpu_layers'.
        """
        super().__init__(model_name, config)  # model_name is the *key*
        if "name" not in config:
            raise ValueError("Ollama config must include a 'name' key with the actual model name.")

        self.ollama_model_name = config["name"]  # Store the *actual* model name
        self.num_gpu_layers = config.get("num_gpu_layers") # Get the new parameter
        self.model_configs = config
        self.llm_model = self._create_ollama_model()

    def _create_ollama_model(self) -> OllamaLLM:
        """Creates and returns an OllamaLLM instance based on the configuration."""
        return OllamaLLM(
            model=self.ollama_model_name,
            temperature=self.temperature,
            num_ctx=self.context_window,
            num_predict=self.num_predict,
            num_gpu=self.num_gpu_layers, # Pass the parameter to the LLM
        )

    def invoke(self, prompt: str) -> str:
        """
        Invokes the Ollama LLM with the given prompt and returns the response.
        Implements retry logic for handling potential connection errors.

        Args:
            prompt (str): The prompt to send to the Ollama model.

        Returns:
            str: The Ollama model's response.
        """
        retries = 2
        wait_times = [5, 25]  # Exponential backoff in seconds

        for attempt in range(retries):
            try:
                return self.llm_model.invoke(prompt)
            except ResponseError as e:
                if attempt < retries - 1:
                    wait_time = wait_times[min(attempt, len(wait_times) - 1)]
                    print(
                        f"Ollama ResponseError: {e}. Retry attempt {attempt + 1}/{retries}. Waiting {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    print(f"Ollama ResponseError: {e}. Max retries reached. Raising error.")
                    raise  # Re-raise the exception after max retries
            except Exception as e: # Catch other potential exceptions
                print(f"Unexpected error during Ollama invoke: {e}")
                raise # Re-raise unexpected exceptions

if __name__ == "__main__":
    # Example usage (assuming Ollama is running and you have llama3.2 model):
    ollama_config = {
        "name": "llama3.2:1B-128k",  # The *actual* Ollama model name
        "temperature": 0.0,
        "num_predict": 256,
        "context_window": 131072,
        "num_gpu_layers": -1, # Example with the new parameter
    }
    # Use a key that might be different from the actual model name
    ollama_connector = OllamaConnector("llama3.2:1B-128k", ollama_config)

    prompt_text = "What is the capital of France?"
    response = ollama_connector.invoke(prompt_text)
    print(f"Prompt: {prompt_text}")
    print(f"Response from Ollama: {response}")
