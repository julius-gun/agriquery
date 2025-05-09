# llm_connectors/gemini_connector.py
# python -m llm_connectors.gemini_connector

import os
import time
import logging
import random
from typing import Dict, Any, List, Optional # Added Optional
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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeminiConnector(BaseLLMConnector):
    """
    Connector for Gemini LLMs. Includes logic to switch API keys on ResourceExhausted errors,
    cycling through the keys if necessary.
    """

    # Define the order of API key environment variable names to try
    API_KEY_ENV_VARS: List[str] = ["GEMINI_API_KEY_DC", "GEMINI_API_KEY_FRIEDM", "GEMINI_API_KEY_SG", "GEMINI_API_KEY_SG2"]
    INVOKE_TIMEOUT_SECONDS: int = 3600 # 60 minutes, adjust as needed
    KEY_SWITCH_BASE_WAIT_SECONDS: int = 40 # Base wait time after a key switch
    KEY_SWITCH_RANDOM_WAIT_SECONDS: int = 20 # Max random seconds to add to base wait

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the GeminiConnector.

        Args:
            model_name (str): The user-friendly name of the Gemini model (e.g., "gemini-pro").
            config (Dict[str, Any]): Configuration parameters for the Gemini model.
                                    Should include 'name' (the internal model name, e.g., "models/gemini-pro"),
                                    and optionally 'temperature', 'max_output_tokens', etc.

        Raises:
            ValueError: If none of the specified API key environment variables are set or configurable.
        """
        super().__init__(model_name, config)
        self.model_configs = config
        self.current_api_key_index: int = -1 # Start with no key selected
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

        # Find and configure the first valid API key
        if not self._configure_api_key(startIndex=0): # Pass only startIndex, stopIndex defaults to None
            raise ValueError(f"Could not find or configure ANY valid Gemini API key: {self.API_KEY_ENV_VARS}")

    def _configure_api_key(self, startIndex: int = 0, stopIndex: Optional[int] = None) -> bool:
        """
        Finds the next available API key from the list, configures genai,
        and recreates the model and chat session. Iterates from startIndex
        up to stopIndex (exclusive) or the end of the list.

        Args:
            startIndex (int): The index in API_KEY_ENV_VARS to start searching from.
            stopIndex (Optional[int]): The index in API_KEY_ENV_VARS to stop searching before.
                                      If None, searches to the end of the list.

        Returns:
            bool: True if a key was successfully configured, False otherwise.
        """
        logger.debug(f"Attempting to configure API key starting from index {startIndex}"
                     f"{f' up to index {stopIndex}' if stopIndex is not None else ' to the end'}.")

        # Determine the end point for the loop
        end_limit = stopIndex if stopIndex is not None else len(self.API_KEY_ENV_VARS)
        if stopIndex is not None and stopIndex <= startIndex:
            # If stopIndex is provided and not after startIndex, the range is empty or invalid in this context.
            logger.debug(f"Stop index ({stopIndex}) is not after start index ({startIndex}). No keys to check in this range.")
            return False

        # Ensure loop range is within bounds
        effective_end = min(end_limit, len(self.API_KEY_ENV_VARS))
        effective_start = max(0, startIndex)

        for i in range(effective_start, effective_end):
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

        logger.debug(f"Finished checking keys in range {effective_start} to {effective_end}. No suitable key found in this range.")
        # Don't reset state here unless *all* keys failed initially during __init__
        if startIndex == 0 and stopIndex is None and self.current_api_key_index == -1: # Check if this is the initial full scan failure
             logger.warning("No valid Gemini API keys found to configure across the entire list during initial setup.")
             # Keep state as it is (no active key)

        return False # No suitable key found or configured in the checked range

    def _try_next_api_key(self) -> bool:
        """
        Attempts to configure the next API key in the list, wrapping around
        to the beginning if the end is reached. Tries the full list once,
        starting after the current key.

        Returns:
            bool: True if a *different* key was successfully configured, False otherwise.
        """
        if not self.API_KEY_ENV_VARS:
            logger.warning("API key list is empty. Cannot switch keys.")
            return False

        num_keys = len(self.API_KEY_ENV_VARS)
        if num_keys <= 1:
            logger.warning("Only one or zero API keys configured. Cannot switch.")
            return False

        # Determine the index of the key that just failed (or -1 if none was active)
        failed_key_index = self.current_api_key_index

        # Calculate the starting index for the search (the one after the failed key, wrapping around)
        start_search_index = (failed_key_index + 1) % num_keys

        logger.warning(f"Rate limit hit on key index {failed_key_index}. Attempting to switch key, starting search from index {start_search_index}.")

        # Iterate through the keys in order, starting from start_search_index,
        # trying each one until we loop back to the original failed key.
        for i in range(num_keys):
            check_index = (start_search_index + i) % num_keys

            # If we've looped back to the key that originally failed, stop searching.
            if check_index == failed_key_index:
                logger.debug(f"Looped back to the originally failed key index {failed_key_index}. Stopping search.")
                break

            logger.debug(f"Attempting to configure next key at index {check_index}...")
            # Try configuring only this specific index
            if self._configure_api_key(startIndex=check_index, stopIndex=check_index + 1):
                 # Check if the configured index is actually different from the failed one
                 # (This check is slightly redundant as we break the loop before re-checking the failed index, but good for clarity)
                 if self.current_api_key_index != failed_key_index:
                     logger.info(f"Successfully switched to API key index {self.current_api_key_index}.")
                     return True
                 else:
                     # This case should ideally not be hit due to the loop break condition
                     logger.warning(f"Configuration reported success but index {self.current_api_key_index} is the same as the failed index {failed_key_index}. Treating as failure to switch.")
                     # Restore failed index? Or just return False? Let's return False.
                     # We might need to restore the failed_key_index state if _configure_api_key changed it inappropriately.
                     self.current_api_key_index = failed_key_index # Restore original failing index
                     return False


        # If we exit the loop, it means no *other* key could be configured.
        logger.warning("Could not configure any *other* API key after a full cycle attempt.")
        # Keep the current_api_key_index as it was (the one that failed),
        # as invoke will backoff and retry with this one if max_retries not reached.
        return False

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
        Includes an initial delay, retries indefinitely on ResourceExhausted errors
        by cycling API keys and using exponential backoff, until an overall timeout
        is reached or another error occurs.
        Each invoke call uses a fresh chat session to prevent history accumulation.

        Args:
            prompt (str): The prompt to send to Gemini.

        Returns:
            str: The Gemini model's response or an error message (timeout or unexpected).
        """
        if not self.API_KEY_ENV_VARS:
             logger.error("GeminiConnector invoke failed: No API Keys configured.")
             return "Error: Gemini connector has no API keys configured."

        DELAY_SECONDS = 11 # Mandatory delay before each *initial* API call attempt
        base_delay = 1     # Base delay in seconds for exponential backoff
        max_backoff_exponent = 10 # Cap backoff exponent (2^10 = 1024s ~ 17 mins)

        start_time = time.time()
        attempt = 0 # Initialize attempt counter for backoff calculation

        # Add initial delay before the first attempt
        logger.debug(f"GeminiConnector: Applying initial delay of {DELAY_SECONDS} seconds before API call...")
        time.sleep(DELAY_SECONDS)

        while True: # Loop indefinitely until success, timeout, or other error
            # --- Check Timeout ---
            elapsed_time = time.time() - start_time
            if elapsed_time > self.INVOKE_TIMEOUT_SECONDS:
                timeout_msg = (
                    f"Error: Timeout ({self.INVOKE_TIMEOUT_SECONDS}s) reached after "
                    f"{attempt} attempts due to persistent API issues (likely rate limiting). "
                    f"Last used key index: {self.current_api_key_index}."
                )
                logger.error(f"GeminiConnector: {timeout_msg}")
                return timeout_msg

            # --- Step 1: Ensure Gemini Model is available and configured ---
            if not self.gemini_model:
                logger.info(f"Invoke (Attempt {attempt + 1}): Gemini model is not initialized. Attempting to configure.")
                current_idx_for_reconfig = self.current_api_key_index if self.current_api_key_index != -1 else 0
                if not self._configure_api_key(startIndex=current_idx_for_reconfig, stopIndex=current_idx_for_reconfig + 1):
                    logger.warning(f"Model configuration with key index {current_idx_for_reconfig} failed. Attempting to cycle to the next key.")
                    if not self._try_next_api_key(): # _try_next_api_key also calls _configure_api_key
                        error_msg = "Error: Gemini model could not be initialized after attempting all API keys."
                        logger.error(f"Invoke: {error_msg}")
                        return error_msg
                    logger.info(f"Invoke: Successfully configured model using new key index {self.current_api_key_index}.")
                else:
                    logger.info(f"Invoke: Successfully configured model using key index {self.current_api_key_index}.")
                # If model configuration succeeded, self.gemini_model is valid and self.chat_session was freshly created.
                # Increment attempt as model configuration is a significant action.
                attempt += 1
                # Continue to Step 2 to ensure chat session is fresh for this specific call,
                # even if _configure_api_key already created one. This makes Step 2 the single source of truth.

            # --- Step 2: Create a fresh Chat Session for this specific invoke call ---
            # This ensures that history from previous, unrelated invoke calls is cleared.
            try:
                if not self.gemini_model:
                    logger.warning(f"Invoke (Attempt {attempt + 1}): gemini_model is None before creating chat session. Retrying model configuration.")
                    attempt += 1 # Count this as part of the ongoing attempt cycle
                    time.sleep(0.5) # Brief pause before retrying the loop to re-trigger Step 1
                    continue

                # logger.debug(f"Invoke (Attempt {attempt + 1}): Creating fresh chat session.")
                self.chat_session = self._create_chat_session() # Ensures statelessness for each invoke
            except RuntimeError as e:
                logger.error(f"Invoke (Attempt {attempt + 1}): Failed to create chat session: {e}. Model might be invalid. Forcing model re-check.")
                self.gemini_model = None # Mark to trigger re-config in Step 1 on next iteration.
                self.chat_session = None
                attempt += 1 # Count this as part of the ongoing attempt cycle
                time.sleep(1) # Small delay before retrying the loop
                continue

            # --- Step 3: Try Sending Message ---
            try:
                current_key_name = "N/A"
                current_key_index = self.current_api_key_index # Cache index for this attempt
                if 0 <= current_key_index < len(self.API_KEY_ENV_VARS):
                     current_key_name = self.API_KEY_ENV_VARS[current_key_index]
                logger.debug(f"GeminiConnector: Sending message (Attempt {attempt + 1}) using key '{current_key_name}' (index {current_key_index})...")

                if not self.chat_session: # Should be extremely rare now due to Step 2
                     logger.error(f"Invoke (Attempt {attempt + 1}): Chat session is unexpectedly None right before send_message. Forcing model re-check.")
                     self.gemini_model = None # Mark model as needing re-init
                     self.chat_session = None
                     attempt += 1 # Count this as part of the ongoing attempt cycle
                     time.sleep(1) # Small delay
                     continue

                response = self.chat_session.send_message(prompt)
                logger.debug(f"GeminiConnector: Received response successfully on attempt {attempt + 1}.")
                return response.text # Success, return response

            # --- Handle Rate Limiting ---
            except google_exceptions.ResourceExhausted as e:
                current_key_name = "N/A"
                current_key_index = self.current_api_key_index # Get index again, might have changed if reconfig happened above
                if 0 <= current_key_index < len(self.API_KEY_ENV_VARS):
                     current_key_name = self.API_KEY_ENV_VARS[current_key_index]
                logger.warning(f"GeminiConnector: Rate limit error (429 Resource Exhausted) on attempt {attempt + 1} with key '{current_key_name}' (index {current_key_index}). {e}")

                # Increment attempt counter *before* deciding next action based on it
                attempt += 1

                # Attempt to switch to the next key (handles wrap-around)
                logger.info("Attempting to switch API key...")
                if self._try_next_api_key():
                    logger.info(f"Successfully switched to API key index {self.current_api_key_index}.")
                    # Introduce delay after successful key switch
                    key_switch_wait_duration = self.KEY_SWITCH_BASE_WAIT_SECONDS + random.uniform(0, self.KEY_SWITCH_RANDOM_WAIT_SECONDS)
                    logger.info(f"Waiting for {key_switch_wait_duration:.2f} seconds after key switch before retrying...")
                    
                    # Check if this wait exceeds remaining timeout
                    current_elapsed_after_switch_attempt = time.time() - start_time
                    if current_elapsed_after_switch_attempt + key_switch_wait_duration > self.INVOKE_TIMEOUT_SECONDS:
                        remaining_time = self.INVOKE_TIMEOUT_SECONDS - current_elapsed_after_switch_attempt
                        if remaining_time > 0:
                            logger.warning(f"Key switch wait time ({key_switch_wait_duration:.2f}s) exceeds remaining timeout. Waiting for remaining {remaining_time:.2f}s.")
                            time.sleep(remaining_time)
                        # The timeout check at the start of the next loop iteration will catch this.
                        continue 
                    else:
                        time.sleep(key_switch_wait_duration)

                    # attempt = 0 # Optional: Reset attempt counter if you want fresh backoff for the new key
                                # Keeping attempts cumulative for overall timeout and general backoff progression
                    continue # Retry with the new key after the delay
                else:
                    # _try_next_api_key returned False - no *other* key could be configured.
                    # Stay with the current key index and apply exponential backoff.
                    logger.warning(f"Failed to switch to a different usable API key after full cycle. Applying backoff and retrying on the current key index {current_key_index}.")

                    # --- Apply Backoff Delay ---
                    # Cap the exponent to prevent excessively long sleeps
                    # Use attempt-1 because attempt was already incremented for this failure
                    capped_exponent = min(attempt - 1, max_backoff_exponent)
                    backoff_time = (base_delay * (2 ** capped_exponent)) + random.uniform(0, 1)
                    logger.warning(f"GeminiConnector: Applying backoff delay of {backoff_time:.2f} seconds (Attempt {attempt})...")

                    # Check if backoff exceeds remaining time
                    current_elapsed = time.time() - start_time # Re-check elapsed time
                    if current_elapsed + backoff_time > self.INVOKE_TIMEOUT_SECONDS:
                         wait_time = self.INVOKE_TIMEOUT_SECONDS - current_elapsed
                         if wait_time > 0:
                              logger.warning(f"Backoff time ({backoff_time:.2f}s) exceeds remaining timeout. Waiting for remaining {wait_time:.2f}s before final timeout check.")
                              time.sleep(wait_time)
                         # The timeout check at the start of the next loop iteration will catch this.
                         continue
                    else:
                         time.sleep(backoff_time)
                         continue # Continue to next iteration after sleeping


            # --- Handle Other Exceptions ---
            except Exception as e:
                # Catch other potential exceptions (e.g., network errors, other API errors)
                logger.error(f"GeminiConnector: An unexpected error occurred during invocation (Attempt {attempt + 1}): {e}", exc_info=True)
                # Fail fast on unexpected errors
                return f"Error: An unexpected error occurred. {e}" # <<< FAILURE: Exit loop and return error

        # This part of the code should theoretically be unreachable because the `while True`
        # loop is only exited by explicit `return` statements (success, timeout, other error).
        # Adding a fallback return for safety.
        logger.error("GeminiConnector: invoke method exited loop unexpectedly.")
        return "Error: Exited invoke loop unexpectedly."


if __name__ == "__main__":
    # Example usage (ensure at least one GEMINI_API_KEY* environment variable is set):
    # Configure logging for example run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

        # --- Test Scenario: Timeout ---
        # To test timeout, you might need to:
        # 1. Set INVOKE_TIMEOUT_SECONDS to a small value (e.g., 15).
        # 2. Ensure all configured API keys will hit rate limits quickly (e.g., by running many requests in parallel).
        # 3. Observe if the "Error: Timeout..." message is returned after the specified duration.
        # GeminiConnector.INVOKE_TIMEOUT_SECONDS = 15 # Temporarily override for testing

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

        # --- Manual Testing Notes ---
        # To test ResourceExhausted + Cycling:
        # 1. Ensure multiple GEMINI_API_KEY* env vars are set (e.g., _DC, _LS, _SG).
        # 2. Make rapid calls (potentially using threads or multiple scripts) to trigger 429s.
        # 3. Observe logs for "Rate limit hit...", "Attempting to switch key...", "Successfully switched...",
        #    and potentially "Wrapping around..." or "Failed to switch... Continuing retry/backoff...".
        # 4. If rate limiting persists across all keys, you should eventually see the "Max retries reached..." error.
        # --------------------------

    except ValueError as ve:
        logger.error(f"Initialization failed: {ve}")
    except Exception as ex:
        logger.error(f"An unexpected error occurred during the example run: {ex}", exc_info=True)