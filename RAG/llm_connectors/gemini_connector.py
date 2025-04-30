# llm_connectors/gemini_connector.py
# python -m llm_connectors.gemini_connector

import os
import time
import logging
import random # Import random for jitter
from typing import Dict, Any
import google.generativeai as genai
# Import the specific exception for rate limiting/quota issues
from google.api_core import exceptions as google_exceptions
from llm_connectors.base_llm_connector import BaseLLMConnector

# Basic logging configuration (can be adjusted or handled globally)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        api_key = os.environ.get("GEMINI_API_KEY")
        # api_key = os.environ.get("HUGGING_FACE_API_KEY")
        # print(f"API key: {api_key}")
        if not api_key:
            raise ValueError("The GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)  # Ensure GEMINI_API_KEY is set in environment


        # Generation config - using values from provided example and config.  Handle potential
        # missing keys gracefully.  Also rename max_tokens to max_output_tokens for Gemini.
        self.generation_config = {
            "temperature": self.temperature,
            "top_p": self.config.get("top_p", 0.0),  # Default values if not in config
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
        Includes a mandatory delay before each invocation.
        and implements retry logic with exponential backoff for rate limit errors (429).


        Args:
            prompt (str): The prompt to send to Gemini.

        Returns:
            str: The Gemini model's response. Handles potential errors including retries.
        """
        DELAY_SECONDS = 8 # Delay in seconds
        logging.debug(f"GeminiConnector: Waiting {DELAY_SECONDS} seconds before API call...") # Use debug level for less noise
        time.sleep(DELAY_SECONDS)
        max_retries = 3 # Maximum number of retries
        base_delay = 5  # Base delay in seconds for backoff

        for attempt in range(max_retries + 1): # +1 to include the initial attempt
            try:
                logging.debug(f"GeminiConnector: Sending message (Attempt {attempt + 1}/{max_retries + 1})...")
                response = self.chat_session.send_message(prompt)
                logging.debug(f"GeminiConnector: Received response successfully on attempt {attempt + 1}.")
                return response.text # Success, return response

            except google_exceptions.ResourceExhausted as e:
                logging.warning(f"GeminiConnector: Rate limit error (429 Resource Exhausted) encountered on attempt {attempt + 1}. {e}")
                if attempt < max_retries:
                    # Calculate backoff time: base_delay * 2^attempt + random_jitter
                    backoff_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    logging.warning(f"GeminiConnector: Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                else:
                    logging.error(f"GeminiConnector: Max retries ({max_retries}) reached after rate limit error. Failing request.")
                    # Re-raise the last exception or return a specific error message
                    # raise # Option 1: Re-raise the exception
                    return f"Error: Max retries reached due to rate limiting. Last error: {e}" # Option 2: Return error string

            except Exception as e:
                # Catch other potential exceptions (e.g., network errors, other API errors)
                logging.error(f"GeminiConnector: An unexpected error occurred during invocation (Attempt {attempt + 1}): {e}", exc_info=True)
                # Decide if retrying makes sense for other errors, here we fail fast
                return f"Error: An unexpected error occurred. {e}" # Return error string for non-retryable errors

        # This part should ideally not be reached if logic is correct, but acts as a fallback.
        logging.error("GeminiConnector: invoke method exited loop unexpectedly.")
        return "Error: Failed after multiple retries or unexpected loop exit."


if __name__ == "__main__":
    # Example usage (assuming GEMINI_API_KEY environment variable is set):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configure logging for example run
    gemini_config = {
        "name": "gemini-2.5-flash-preview-04-17",  # Use the *internal* Gemini model name
        "temperature": 0.0,
        "num_predict": 256,
        "top_k": 40,  # Example additional parameter
        "top_p": 0.8,  # Example additional parameter
    }
    gemini_connector = GeminiConnector("gemini-2.5-flash-preview-04-17", gemini_config)

    prompt_text = "What is the capital of Spain?"
    logging.info(f"Sending first prompt: {prompt_text}")
    # Simulate a rate limit error on the first call for testing (requires mocking)
    # To test manually, you might need to trigger the rate limit naturally.
    response = gemini_connector.invoke(prompt_text)
    print(f"Prompt: {prompt_text}")
    print(f"Response from Gemini: {response}")

    # Example of error handling:
    bad_prompt = "Generate an extremely long response that will exceed the token limit."  # This might cause an error
    error_response = gemini_connector.invoke(bad_prompt)
    print(f"\nPrompt that might cause an error: {bad_prompt}")
    print(f"Response: {error_response}")  # Will print the error message