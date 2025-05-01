# llm_connectors/gemini_connector.py
# python -m llm_connectors.gemini_connector

import os
import time
import logging
import random
from typing import Dict, Any, List
import google.generativeai as genai
# Import the specific exception for rate limiting/quota issues
from google.api_core import exceptions as google_exceptions
from llm_connectors.base_llm_connector import BaseLLMConnector

# Basic logging configuration (can be adjusted or handled globally)
# Consider moving configuration outside the class if used globally
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# Using basicConfig here might interfere if logging is configured elsewhere.
# It's often better to get a logger instance:
logger = logging.getLogger(__name__)
# Ensure a handler is added if run as a script or no other config exists
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class GeminiConnector(BaseLLMConnector):
    """
    Connector for Gemini LLMs. Includes logic to switch API keys on ResourceExhausted errors.
    """

    # Define the order of API key environment variable names to try
    API_KEY_ENV_VARS: List[str] = ["GEMINI_API_KEY_DC", "GEMINI_API_KEY_LS", "GEMINI_API_KEY_SG"]

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the GeminiConnector.

        Args:
            model_name (str): The user-friendly name of the Gemini model (e.g., "gemini-pro").
            config (Dict[str, Any]): Configuration parameters for the Gemini model.
                                    Should include 'name' (the internal model name, e.g., "models/gemini-pro"),
                                    and optionally 'temperature', 'max_output_tokens', etc.

        Raises:
            ValueError: If none of the specified API key environment variables are set.
        """
        super().__init__(model_name, config)
        self.model_configs = config
        self.current_api_key_index: int = -1
        self.active_api_key: str | None = None
        self.gemini_model: genai.GenerativeModel | None = None
        self.chat_session: genai.ChatSession | None = None

        # Initialize generation config first (doesn't depend on API key)
        self.generation_config = {
            "temperature": self.temperature,
            "top_p": self.config.get("top_p", 0.0),  # Default values if not in config
            "top_k": self.config.get("top_k", 64),
            "max_output_tokens": self.num_predict,
            "response_mime_type": "text/plain", # Keep if specifically needed, otherwise omit
        }

        # Find and configure the first valid API key and initialize model/session
        if not self._configure_api_key(startIndex=0):
            raise ValueError(f"Could not find or configure any of the specified Gemini API keys in environment variables: {self.API_KEY_ENV_VARS}")

    def _configure_api_key(self, startIndex: int = 0) -> bool:
        """
        Finds the next available API key from the list, configures genai,
        and recreates the model and chat session.

        Args:
            startIndex (int): The index in API_KEY_ENV_VARS to start searching from.

        Returns:
            bool: True if a key was successfully configured, False otherwise.
        """
        logger.debug(f"Attempting to configure API key starting from index {startIndex}.")
        for i in range(startIndex, len(self.API_KEY_ENV_VARS)):
            key_name = self.API_KEY_ENV_VARS[i]
            api_key = os.environ.get(key_name)

            if api_key:
                logger.info(f"Attempting to configure Gemini with API key from '{key_name}' (index {i}).")
                try:
                    # Configure the genai library with the found key
                    genai.configure(api_key=api_key)

                    # Re-create the generative model instance
                    # Use the *internal* model name from the config
                    self.gemini_model = genai.GenerativeModel(
                        model_name=self.config["name"], # Use the "name" field from the config
                        generation_config=self.generation_config,
                        # System instruction could be part of config or set here
                        system_instruction=self.config.get("system_instruction", "Answer the question based on the context provided."),
                    )

                    # Re-create the chat session
                    self.chat_session = self._create_chat_session()

                    # Update state if successful
                    self.current_api_key_index = i
                    self.active_api_key = api_key
                    logger.info(f"Successfully configured Gemini with API key from '{key_name}'.")
                    return True # Configuration successful

                except Exception as e:
                    logger.error(f"Failed to configure Gemini or create model/session with key from '{key_name}': {e}", exc_info=True)
                    # Continue to the next key if configuration fails
            else:
                logger.debug(f"API key environment variable '{key_name}' not found or is empty.")

        logger.warning("No more valid Gemini API keys found to configure.")
        self.current_api_key_index = -1 # Indicate no key is active if all fail
        self.active_api_key = None
        self.gemini_model = None
        self.chat_session = None
        return False # No suitable key found or configured

    def _try_next_api_key(self) -> bool:
        """Attempts to configure the next API key in the list."""
        next_index = self.current_api_key_index + 1
        if next_index < len(self.API_KEY_ENV_VARS):
            logger.warning(f"Attempting to switch to next API key (index {next_index}).")
            # Attempt to configure starting from the next index
            return self._configure_api_key(startIndex=next_index)
        else:
            logger.warning("No more API keys available in the list to try.")
            return False # No more keys to try

    def _create_chat_session(self):
        """Creates a new chat session for Gemini."""
        if not self.gemini_model:
             logger.error("Cannot create chat session: Gemini model not initialized.")
             # Depending on desired robustness, could raise error or return None
             raise RuntimeError("Gemini model not initialized before creating chat session.")
        # Start a new chat session. History might be managed differently depending on use case.
        # For simple invoke, starting fresh each time might be intended.
        return self.gemini_model.start_chat(history=[])


    def invoke(self, prompt: str) -> str:
        """
        Invokes the Gemini LLM with the given prompt and returns the response.
        Includes a delay, retries on ResourceExhausted errors, and attempts
        to switch API keys if ResourceExhausted occurs.

        Args:
            prompt (str): The prompt to send to Gemini.

        Returns:
            str: The Gemini model's response or an error message.
        """
        if not self.chat_session:
             logger.error("GeminiConnector invoke failed: Chat session not available.")
             return "Error: Gemini chat session is not initialized."

        DELAY_SECONDS = 11 # Mandatory delay before each *initial* API call attempt
        max_retries = 3 # Maximum number of retries *per API key* potentially
        base_delay = 1 # Base delay in seconds for exponential backoff (starts low)

        keys_exhausted_this_call = False # Flag specific to this invoke call

        # Add initial delay before the first attempt
        logger.debug(f"GeminiConnector: Applying initial delay of {DELAY_SECONDS} seconds before API call...")
        time.sleep(DELAY_SECONDS)

        for attempt in range(max_retries + 1): # +1 to include the initial attempt
            try:
                current_key_name = self.API_KEY_ENV_VARS[self.current_api_key_index] if self.current_api_key_index != -1 else "N/A"
                logger.debug(f"GeminiConnector: Sending message (Attempt {attempt + 1}/{max_retries + 1}) using key '{current_key_name}' (index {self.current_api_key_index})...")

                # Ensure chat session is still valid (might have been reset by key switch)
                if not self.chat_session:
                     logger.error("Invoke failed: Chat session became unavailable unexpectedly.")
                     return "Error: Chat session lost during execution."

                response = self.chat_session.send_message(prompt)
                logger.debug(f"GeminiConnector: Received response successfully on attempt {attempt + 1}.")
                return response.text # Success, return response

            except google_exceptions.ResourceExhausted as e:
                current_key_name = self.API_KEY_ENV_VARS[self.current_api_key_index] if self.current_api_key_index != -1 else "N/A"
                logger.warning(f"GeminiConnector: Rate limit error (429 Resource Exhausted) on attempt {attempt + 1} with key '{current_key_name}'. {e}")

                # Check if we have already tried all keys during this invoke call
                if keys_exhausted_this_call:
                    logger.warning("All API keys previously tried for this call resulted in ResourceExhausted. Continuing backoff on last key.")
                else:
                    # Attempt to switch to the next key
                    if self._try_next_api_key():
                        logger.info(f"Successfully switched to API key index {self.current_api_key_index}. Retrying immediately with the new key.")
                        # Reset attempt count for the new key? Debatable. Let's reset for simplicity.
                        # attempt = -1 # This would restart the loop for the new key, maybe too complex.
                        # Let's just continue the loop. The next iteration will use the new key.
                        # No backoff needed yet, as we just switched keys.
                        continue # Skip backoff and retry immediately with the new key configuration
                    else:
                        # _try_next_api_key returned False, meaning no more keys are available.
                        keys_exhausted_this_call = True # Mark that we've run out of keys for this invoke() call
                        logger.warning("All available API keys have been tried and failed with ResourceExhausted. Continuing retry/backoff on the last used key.")

                # If we are here, either all keys were exhausted, or the key switch failed,
                # or we chose not to switch keys for other reasons. Proceed with backoff.
                if attempt < max_retries:
                    # Calculate backoff time: base_delay * 2^attempt + random_jitter
                    backoff_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    logger.warning(f"GeminiConnector: Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                else:
                    # Max retries reached for the *last attempted key*
                    final_error_msg = f"Error: Max retries ({max_retries}) reached due to rate limiting."
                    if keys_exhausted_this_call:
                        final_error_msg += " All API keys were tried and exhausted."
                    else:
                         final_error_msg += f" Last attempt was with key '{current_key_name}'."
                    final_error_msg += f" Last error: {e}"
                    logger.error(f"GeminiConnector: {final_error_msg}")
                    return final_error_msg # Return specific error message

            except Exception as e:
                # Catch other potential exceptions (e.g., network errors, other API errors)
                logger.error(f"GeminiConnector: An unexpected error occurred during invocation (Attempt {attempt + 1}): {e}", exc_info=True)
                # Decide if retrying makes sense for other errors, here we fail fast
                return f"Error: An unexpected error occurred. {e}" # Return error string for non-retryable errors

        # This part should ideally not be reached if logic is correct, but acts as a fallback.
        logger.error("GeminiConnector: invoke method exited loop unexpectedly.")
        return "Error: Failed after multiple retries or unexpected loop exit."


if __name__ == "__main__":
    # Example usage (ensure at least one GEMINI_API_KEY* environment variable is set):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configure logging for example run
    # Ensure logger level is appropriate if using getLogger approach
    # logging.getLogger(__name__).setLevel(logging.INFO)
    # logging.getLogger().addHandler(logging.StreamHandler()) # Add handler if needed

    gemini_config = {
        # Make sure this internal name matches the key type (e.g., Pro vs Flash)
        "name": "models/gemini-1.5-flash-latest",
        "temperature": 0.1, # Low temperature for factual tasks
        "num_predict": 512, # Increased prediction length
        "top_k": 40,
        "top_p": 0.95,
        "system_instruction": "You are a helpful assistant providing concise answers."
    }

    try:
        # Use a user-friendly name, the internal name is in the config
        gemini_connector = GeminiConnector("gemini-flash", gemini_config)

        prompt_text = "What is the distance between the Earth and the Moon in kilometers?"
        logger.info(f"Sending first prompt: {prompt_text}")

        # --- Manual Testing Notes ---
        # To test ResourceExhausted: Make many rapid calls (might require multiple scripts/threads)
        # To test key switching: Set GEMINI_API_KEY to an invalid or exhausted key,
        #                       and set GEMINI_API_KEY_3/2 to a valid key.
        # To test missing keys: Unset all GEMINI_API_KEY* variables (should raise ValueError on init).
        # --------------------------

        response = gemini_connector.invoke(prompt_text)
        print("-" * 20)
        print(f"Prompt: {prompt_text}")
        print(f"Response from Gemini: {response}")
        print("-" * 20)

        # Example of a follow-up or different prompt
        prompt_text_2 = "Who wrote 'The Hobbit'?"
        logger.info(f"Sending second prompt: {prompt_text_2}")
        response_2 = gemini_connector.invoke(prompt_text_2)
        print(f"Prompt: {prompt_text_2}")
        print(f"Response from Gemini: {response_2}")
        print("-" * 20)

    except ValueError as ve:
        logger.error(f"Initialization failed: {ve}")
    except Exception as ex:
        logger.error(f"An unexpected error occurred during the example run: {ex}", exc_info=True)