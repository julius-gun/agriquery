# main.py
import time
import sys
import os

# --- Configuration ---
# You might want to add argument parsing here later if needed
# For now, it assumes rag_tester uses the default "config.json"

# --- Import the main testing function ---
# This assumes main.py is in the root directory alongside rag_tester.py
try:
    # Ensure the current directory is in the Python path if needed,
    # though usually not necessary if run from the project root.
    # sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    # run main() of utils.create_databases.py 
    from utils import create_databases
    from rag_tester import run_rag_test
    # Import the module containing the database creation logic
    print("Successfully imported 'run_rag_test' and 'utils.create_databases'")
except ImportError as e:
    print(f"Error: Could not import required modules ('run_rag_test', 'utils.create_databases').")
    print(f"Ensure 'main.py' is in the correct directory relative to 'rag_tester.py' and the 'utils' folder.")
    print(f"and that the project structure allows the imports.")
    print(f"Original error: {e}")
    sys.exit(1) # Exit if core functions/modules cannot be imported

# --- Main Execution ---
def main():
    """
    Main entry point to create databases and then start the RAG testing process.
    """
    print("\n=============================================")
    print(" main.py: Starting Execution")
    print("=============================================")

    overall_start_time = time.time()

    try:
        # # --- Step 1: Create/Update Databases ---
        # print("\n--- Running Database Creation/Update ---")
        # db_start_time = time.time()
        # # Call the main function from create_databases.py
        # create_databases.main()
        # db_end_time = time.time()
        # print(f"--- Database Creation/Update Finished (Duration: {db_end_time - db_start_time:.2f} seconds) ---")

        # --- Step 2: Run RAG Tests ---
        print("\n--- Initiating RAG Testing Process ---")
        test_start_time = time.time()
        # Call the function that contains the core testing logic
        # It will use the settings defined in config.json and the created databases
        run_rag_test()
        test_end_time = time.time()
        print(f"--- RAG Testing Process Finished (Duration: {test_end_time - test_start_time:.2f} seconds) ---")

        overall_end_time = time.time()
        duration = overall_end_time - overall_start_time
        print("-" * 45)
        print(f"main.py: All steps completed successfully.")
        print(f"Total execution time recorded by main.py: {duration:.2f} seconds.")
        print("=============================================")

    except Exception as e:
        # Catch potential exceptions during database creation or test run
        overall_end_time = time.time()
        duration = overall_end_time - overall_start_time
        print("-" * 45)
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"main.py: An error occurred during execution!")
        # Log the exception details
        import traceback
        print("\n--- Error Details ---")
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