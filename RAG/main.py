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

    from rag_tester import run_rag_test
    print("Successfully imported 'run_rag_test' from rag_tester.py")
except ImportError as e:
    print(f"Error: Could not import 'run_rag_test' from 'rag_tester.py'.")
    print(f"Ensure 'main.py' is in the correct directory relative to 'rag_tester.py'")
    print(f"and that the project structure allows the import.")
    print(f"Original error: {e}")
    sys.exit(1) # Exit if the core function cannot be imported

# --- Main Execution ---
def main():
    """
    Main entry point to start the RAG testing process.
    """
    print("\n=============================================")
    print(" main.py: Initiating RAG Testing Process")
    print("=============================================")
    print("Executing the main testing logic from rag_tester.py...")
    print("-" * 45)

    start_time = time.time()

    try:
        # Call the function that contains the core testing logic
        # It will use the settings defined in config.json
        run_rag_test()

        end_time = time.time()
        duration = end_time - start_time
        print("-" * 45)
        print(f"main.py: RAG testing process (via run_rag_test) completed.")
        print(f"Total execution time recorded by main.py: {duration:.2f} seconds.")
        print("=============================================")

    except Exception as e:
        # Catch potential exceptions during the test run initiated by run_rag_test
        end_time = time.time()
        duration = end_time - start_time
        print("-" * 45)
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"main.py: An error occurred during the RAG testing process!")
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