# llm_connectors/gemini_connector.py
# python -m llm_connectors.gemini_connector

import os
from typing import Dict, Any
import google.generativeai as genai
from llm_connectors.base_llm_connector import BaseLLMConnector


class GeminiConnector(BaseLLMConnector):
    """Connector for Gemini LLMs."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the GeminiConnector.

        Args:
            model_name (str): The name of the Gemini model to use.  This is the *short* name
                used by the user, e.g., "gemini-pro".  The actual model name used internally
                can be different (and is specified in the config).
            config (Dict[str, Any]): Configuration parameters for the Gemini model.
                                    Should include 'name' (the *internal* model name, e.g., "models/gemini-pro"),
                                    and optionally 'temperature', 'max_output_tokens', 'top_k', 'top_p'.
        """
        super().__init__(model_name, config)  # Call BaseLLMConnector's __init__ to set model_name
        self.model_configs = config 

        # Initialize Gemini API
        # Key change: Get the API key directly here and handle potential missing key.
        api_key = os.environ.get("GEMINI_API_KEY1")
        # api_key = os.environ.get("HUGGING_FACE_API_KEY")
        # print(f"API key: {api_key}")
        if not api_key:
            raise ValueError("The GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)  # Ensure GEMINI_API_KEY is set in environment


        # Generation config - using values from provided example and config.  Handle potential
        # missing keys gracefully.  Also rename max_tokens to max_output_tokens for Gemini.
        self.generation_config = {
            "temperature": self.temperature,
            "top_p": self.config.get("top_p", 0.95),  # Default values if not in config
            "top_k": self.config.get("top_k", 64),
            "max_output_tokens": self.num_predict,  # Use max_tokens from BaseLLMConnector
            "response_mime_type": "text/plain",  # This might not be directly supported, but we keep it for consistency
        }

        # Use the *internal* model name from the config, not the user-provided one.
        self.gemini_model = genai.GenerativeModel(
            model_name=self.config["name"],  # Use the "name" field from the config
            generation_config=self.generation_config,
            system_instruction="Answer the question based on the context provided.", # Added system instruction
        )
        self.chat_session = self._create_chat_session()


    def _create_chat_session(self):
        """Creates a new chat session for Gemini."""
        return self.gemini_model.start_chat(history=[])


    def invoke(self, prompt: str) -> str:
        """
        Invokes the Gemini LLM with the given prompt and returns the response.

        Args:
            prompt (str): The prompt to send to Gemini.

        Returns:
            str: The Gemini model's response.  Handles potential errors.
        """
        try:
            response = self.chat_session.send_message(prompt)
            return response.text
        except Exception as e:
            print(f"Error during Gemini invocation: {e}")
            return f"Error: {e}"  # Return the error message as a string


if __name__ == "__main__":
    # Example usage (assuming GEMINI_API_KEY environment variable is set):
    gemini_config = {
        "name": "gemini-2.0-flash-thinking-exp-01-21",  # Use the *internal* Gemini model name
        "temperature": 0.0,
        "num_predict": 256,
        "top_k": 40,  # Example additional parameter
        "top_p": 0.8,  # Example additional parameter
    }
    gemini_connector = GeminiConnector("gemini-pro", gemini_config)  # User-facing name

    prompt_text = "What is the capital of Spain?"
    response = gemini_connector.invoke(prompt_text)
    print(f"Prompt: {prompt_text}")
    print(f"Response from Gemini: {response}")

    # Example of error handling:
    bad_prompt = "Generate an extremely long response that will exceed the token limit."  # This might cause an error
    error_response = gemini_connector.invoke(bad_prompt)
    print(f"\nPrompt that might cause an error: {bad_prompt}")
    print(f"Response: {error_response}")  # Will print the error message