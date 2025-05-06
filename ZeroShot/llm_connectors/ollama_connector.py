# llm_connectors/ollama_connector.py
import time
import httpx # Import httpx for TimeoutException
from typing import Dict, Any, Optional # Add Optional
from langchain_ollama.llms import OllamaLLM
from llm_connectors.base_llm_connector import BaseLLMConnector
from ollama._types import ResponseError


class OllamaConnector(BaseLLMConnector):
    """Connector for Ollama LLMs with built-in timeout/retry."""

    # Timeout sequence in seconds: 3 min, 5 min, 7 min
    TIMEOUT_SEQUENCE_SECONDS = [180, 300, 420]

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the OllamaConnector.

        Args:
            model_name (str): The *key* of the model in the config (can be a user-defined alias).
            config (Dict[str, Any]): Configuration parameters for the Ollama model.
                                    Should include 'name' (the *actual* Ollama model name),
                                    and optionally 'temperature', 'num_predict', 'context_window'.
                                    Timeout/retry settings from config are ignored.
        """
        super().__init__(model_name, config)  # model_name is the *key*
        # config["name"] is the *actual* Ollama model name.
        if "name" not in config:
            raise ValueError("Ollama config must include a 'name' key with the actual model name.")

        self.ollama_model_name = config["name"]  # Store the *actual* model name
        self.model_configs = config
        # Note: config-based timeout/retry parameters are intentionally ignored.
        # The TIMEOUT_SEQUENCE_SECONDS defines the behavior.

        # We don't create the model here, it will be created with the
        # specific timeout needed for each attempt in the invoke method.
        self.llm_model = None

    def _create_ollama_model(self, timeout: Optional[float] = None) -> OllamaLLM:
        """Creates and returns an OllamaLLM instance with a specific timeout."""
        # Use self.config['name'] to get the *actual* Ollama model name.
        # print(f"DEBUG: Creating OllamaLLM instance for '{self.ollama_model_name}' with timeout: {timeout} seconds")
        return OllamaLLM(
            model=self.ollama_model_name,  # Use the actual Ollama model name.
            temperature=self.temperature,
            num_ctx=self.context_window,
            num_predict=self.num_predict,
            request_timeout=timeout, # Pass the specific timeout for this instance
        )

    def invoke(self, prompt: str) -> str:
        """
        Invokes the Ollama LLM with the given prompt and returns the response.
        Implements retry logic with increasing timeouts.

        Args:
            prompt (str): The prompt to send to the Ollama model.

        Returns:
            str: The Ollama model's response.

        Raises:
            TimeoutError: If all retry attempts time out.
            ResponseError: If a non-timeout Ollama error occurs.
            Exception: For other unexpected errors.
        """
        max_attempts = len(self.TIMEOUT_SEQUENCE_SECONDS)
        last_exception = None

        for attempt in range(max_attempts):
            current_timeout = self.TIMEOUT_SEQUENCE_SECONDS[attempt]
            print(
                f"Attempt {attempt + 1}/{max_attempts}: Invoking Ollama model '{self.ollama_model_name}' "
                f"with timeout {current_timeout} seconds..."
            )

            try:
                # Create a new model instance with the specific timeout for this attempt
                self.llm_model = self._create_ollama_model(timeout=float(current_timeout)) # httpx expects float

                # Invoke the model
                start_time = time.time()
                response = self.llm_model.invoke(prompt)
                end_time = time.time()
                duration = end_time - start_time
                print(f"Attempt {attempt + 1} successful in {duration:.2f} seconds.")
                return response # Success! Return the response.

            except (httpx.TimeoutException, TimeoutError) as e:
                # This catches timeouts from the httpx library used by ollama-python
                last_exception = e
                print(f"Attempt {attempt + 1} timed out after {current_timeout} seconds.")
                if attempt < max_attempts - 1:
                    print("Retrying with longer timeout...")
                else:
                    print(f"Max retries ({max_attempts}) reached after timeout.")
                # Loop continues to the next attempt (if any)

            except ResponseError as e:
                # Handle specific Ollama errors (e.g., connection error, model not found)
                # These are likely not recoverable by retrying with a longer timeout.
                print(f"Ollama ResponseError on attempt {attempt + 1}: {e}. Aborting retries.")
                raise e # Re-raise immediately

            except Exception as e:
                # Catch any other unexpected errors during invocation
                print(f"Unexpected error during Ollama invoke on attempt {attempt + 1}: {e}. Aborting retries.")
                raise e # Re-raise immediately

        # If the loop completes without returning, all attempts timed out
        error_message = (
            f"Failed to get response from Ollama model '{self.ollama_model_name}' "
            f"after {max_attempts} attempts with timeouts {self.TIMEOUT_SEQUENCE_SECONDS} seconds. "
            f"Last error: {last_exception}"
        )
        print(error_message)
        # Raise a TimeoutError to signal the failure condition clearly
        raise TimeoutError(error_message) from last_exception


if __name__ == "__main__":
    # Example usage (assuming Ollama is running and you have llama3.2 model):
    ollama_config = {
        "name": "llama3.2:1B-128k",  # The *actual* Ollama model name
        "temperature": 0.0,
        "num_predict": 256,
        "context_window": 131072,
        # No timeout/retry settings needed here anymore
    }
    # Use a key that might be different from the actual model name
    ollama_connector = OllamaConnector("llama3.2:1B-128k", ollama_config)

    prompt_text = "What is the capital of France?"
    try:
        response = ollama_connector.invoke(prompt_text)
        print(f"\nPrompt: {prompt_text}")
        print(f"Response from Ollama: {response}")
    except Exception as e:
        print(f"\nInvocation failed after retries: {e}")

    # Example prompt designed to potentially take longer (adjust as needed for testing)
    long_prompt = "Write a very detailed analysis of the geopolitical implications of climate change, covering economic, social, and political factors across at least 5 different regions of the world. Ensure the analysis is nuanced and well-supported."
    print("\nAttempting potentially long-running prompt...")
    try:
        response = ollama_connector.invoke(long_prompt)
        print(f"\nPrompt: {long_prompt[:100]}...")
        print(f"Response from Ollama: {response}")
    except Exception as e:
        print(f"\nInvocation failed after retries: {e}")