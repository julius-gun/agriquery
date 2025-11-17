import os
import sys
import json
import time

# --- Path Setup ---
# Add the project's root directory (RAG2_COMPAG) to the Python path
# This allows the script to import modules from the project (e.g., llm_connectors)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from llm_connectors.ollama_connector import OllamaConnector
except ImportError as e:
    print(f"FATAL: Could not import OllamaConnector. Ensure the script is in a 'scripts' subdirectory of your main project.")
    print(f"Project root calculated as: {project_root}")
    print(f"Error details: {e}")
    sys.exit(1)

# --- Configuration ---
CONFIG_FILE_PATH = os.path.join(project_root, 'config.json')
SIMPLE_PROMPT = "Hello! In one short sentence, who are you?"

def test_ollama_models():
    """
    Loads the configuration, iterates through all defined Ollama models,
    and performs a simple invocation test on each one.
    """
    print("--- Starting Ollama Model Health Check ---")
    print(f"Loading configuration from: {CONFIG_FILE_PATH}\n")

    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{CONFIG_FILE_PATH}'.")
        return
    except json.JSONDecodeError:
        print(f"FATAL: Could not parse '{CONFIG_FILE_PATH}'. Please check for JSON errors.")
        return

    ollama_models_config = config.get("llm_models", {}).get("ollama")

    if not ollama_models_config:
        print("INFO: No Ollama models found in the 'llm_models.ollama' section of the config file.")
        return

    total_models = len(ollama_models_config)
    print(f"Found {total_models} Ollama models to test.\n")

    successful_models = []
    failed_models = []

    for i, (model_alias, model_params) in enumerate(ollama_models_config.items()):
        actual_model_name = model_params.get("name", "N/A")
        print(f"--- ({i+1}/{total_models}) Testing Model: '{model_alias}' (Ollama name: '{actual_model_name}') ---")

        try:
            start_time = time.time()
            # The first argument to the connector is the user-defined alias/key
            # from the config, the second is the dictionary of its parameters.
            connector = OllamaConnector(model_alias, model_params)
            
            print(f"  Prompt: '{SIMPLE_PROMPT}'")
            response = connector.invoke(SIMPLE_PROMPT)
            end_time = time.time()
            
            duration = end_time - start_time
            print("\n  ✅ SUCCESS!")
            print(f"  Response: {response.strip()}")
            print(f"  Duration: {duration:.2f} seconds")
            successful_models.append(model_alias)

        except Exception as e:
            print(f"\n  ❌ FAILURE!")
            print(f"  An error occurred: {e}")
            failed_models.append(model_alias)
        
        print("-" * 50 + "\n")

    # --- Print Summary ---
    print("\n================= TEST SUMMARY =================")
    print(f"Total Models Tested: {total_models}")
    print(f"Successful: {len(successful_models)}")
    print(f"Failed: {len(failed_models)}")
    
    if successful_models:
        print("\n✅ Successful Models:")
        for model in successful_models:
            print(f"  - {model}")
            
    if failed_models:
        print("\n❌ Failed Models:")
        for model in failed_models:
            print(f"  - {model}")
    print("==============================================")


if __name__ == "__main__":
    test_ollama_models()
