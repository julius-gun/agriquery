# main.py
import time
import sys
import os

# --- Configuration ---
# Define the config file path (could be made dynamic later)
# CONFIG_FILE = "config.json"
CONFIG_FILE = "config_fast.json"

# --- Import the main testing function ---
# This assumes main.py is in the root directory alongside rag_tester.py
try:
    # Ensure the current directory is in the Python path if needed
    # sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    # run main() of utils.create_databases.py 
    from utils import create_databases
    # Import the NEW entry point function from rag_tester
    from rag_tester import start_rag_tests
    print("Successfully imported 'start_rag_tests' and 'utils.create_databases'")
except ImportError as e:
    print(f"Error: Could not import required modules ('start_rag_tests', 'utils.create_databases').")
    print(f"Ensure 'main.py' is in the correct directory relative to 'rag_tester.py' and the 'utils' folder.")
    print(f"and that the project structure allows the imports.")
    print(f"Original error: {e}")
    sys.exit(1) # Exit if core functions/modules cannot be imported

# --- Main Execution ---
def main():
    """
    Main entry point to potentially create databases and then start the RAG testing process.
    """
    print("\n=============================================")
    print(" main.py: Starting Execution")
    print("=============================================")

    overall_start_time = time.time()

    try:
        # --- Step 1: Create/Update Databases (Optional - uncomment if needed) ---
        print("\n--- Running Database Creation/Update ---")
        db_start_time = time.time()
        # Call the main function from create_databases.py
        # Ensure create_databases.main() also uses the correct config if needed
        create_databases.main(config_path=CONFIG_FILE) # Pass config if needed by create_databases
        db_end_time = time.time()
        print(f"--- Database Creation/Update Finished (Duration: {db_end_time - db_start_time:.2f} seconds) ---")

        # --- Step 2: Run RAG Tests ---
        print("\n--- Initiating RAG Testing Process ---")
        test_start_time = time.time()
        # Call the NEW entry point function, passing the config path
        start_rag_tests(config_path=CONFIG_FILE)
        test_end_time = time.time()
        # Note: Duration calculation here might be slightly less accurate if start_rag_tests
        # includes significant setup time before its internal timing starts.
        print(f"--- RAG Testing Process Finished (Duration recorded by main.py: {test_end_time - test_start_time:.2f} seconds) ---")

        overall_end_time = time.time()
        duration = overall_end_time - overall_start_time
        print("-" * 45)
        print(f"main.py: All steps completed successfully.")
        print(f"Total execution time recorded by main.py: {duration:.2f} seconds.")
        print("=============================================")

    except Exception as e:
        # Catch potential exceptions during database creation or test run
        # (Errors from start_rag_tests will be re-raised and caught here)
        overall_end_time = time.time()
        duration = overall_end_time - overall_start_time
        print("-" * 45)
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"main.py: An error occurred during execution!")
        # Log the exception details
        import traceback
        print("\n--- Error Details (Caught by main.py) ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("---------------------\n")
        print(f"Execution halted after {duration:.2f} seconds due to the error.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1) # Exit with a non-zero code to indicate failure

if __name__ == "__main__":
    # This ensures the main function is called only when the script is executed directly
    main()