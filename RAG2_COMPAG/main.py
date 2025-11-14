# main.py
import time
import sys
import os

# Disable Hugging Face telemetry
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# --- Configuration ---
# Define the config file path (could be made dynamic later)
CONFIG_FILE = "config.json"
# CONFIG_FILE = "config_fast.json"

# --- Import the main testing function ---
# This assumes main.py is in the root directory alongside rag_tester.py
try:
    # Ensure the current directory is in the Python path if needed
    # sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    # run main() of utils.create_databases.py
    from utils import create_databases
    # Import the NEW entry point function from rag_tester
    from rag_tester import start_rag_tests
    # Import components needed for central model initialization
    from utils.config_loader import ConfigLoader
    from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
    print("Successfully imported 'start_rag_tests', 'utils.create_databases', and model components.")
except ImportError as e:
    print("Error: Could not import required modules.")
    print("Ensure 'main.py' is in the correct directory relative to 'rag_tester.py' and the 'utils' folder.")
    print("and that the project structure allows the imports.")
    print(f"Original error: {e}")
    sys.exit(1) # Exit if core functions/modules cannot be imported

# --- Main Execution ---
def main():
    """
    Main entry point to initialize a shared embedding model, potentially create
    databases, and then start the RAG testing process.
    """
    print("\n=============================================")
    print(" main.py: Starting Execution")
    print("=============================================")

    overall_start_time = time.time()

    try:
        # --- Step 1: Centralized Model Initialization ---
        print("\n--- Initializing Shared Embedding Model ---")
        model_init_start_time = time.time()

        # Load config to get model details
        config_loader = ConfigLoader(config_path=CONFIG_FILE)
        embedding_model_config = config_loader.get_embedding_model_config()

        # Create the single, shared instance of the embedding retriever
        shared_embedding_retriever = EmbeddingRetriever(model_config=embedding_model_config)

        model_init_end_time = time.time()
        print(f"--- Shared Embedding Model Initialized (Duration: {model_init_end_time - model_init_start_time:.2f} seconds) ---")


        # --- Step 2: Create/Update Databases ---
        print("\n--- Running Database Creation/Update ---")
        db_start_time = time.time()
        # Pass the config path AND the shared model instance
        create_databases.main(
            config_path=CONFIG_FILE,
            embedding_retriever=shared_embedding_retriever
        )
        db_end_time = time.time()
        print(f"--- Database Creation/Update Finished (Duration: {db_end_time - db_start_time:.2f} seconds) ---")

        # --- Step 3: Run RAG Tests ---
        print("\n--- Initiating RAG Testing Process ---")
        test_start_time = time.time()
        # Pass the config path AND the shared model instance
        start_rag_tests(
            config_path=CONFIG_FILE,
            embedding_retriever=shared_embedding_retriever
        )
        test_end_time = time.time()
        # Note: Duration calculation here might be slightly less accurate if start_rag_tests
        # includes significant setup time before its internal timing starts.
        print(f"--- RAG Testing Process Finished (Duration recorded by main.py: {test_end_time - test_start_time:.2f} seconds) ---")

        overall_end_time = time.time()
        duration = overall_end_time - overall_start_time
        print("-" * 45)
        print("main.py: All steps completed successfully.")
        print(f"Total execution time recorded by main.py: {duration:.2f} seconds.")
        print("=============================================")

    except Exception as e:
        # Catch potential exceptions during model init, database creation, or test run
        # (Errors from start_rag_tests will be re-raised and caught here)
        overall_end_time = time.time()
        duration = overall_end_time - overall_start_time
        print("-" * 45)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("main.py: An error occurred during execution!")
        # Log the exception details
        import traceback
        print("\n--- Error Details (Caught by main.py) ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("---------------------\n")
        print(f"Execution halted after {duration:.2f} seconds due to the error.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1) # Exit with a non-zero code to indicate failure

if __name__ == "__main__":
    # This ensures the main function is called only when the script is executed directly
    main()
