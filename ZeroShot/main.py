# import subprocess
# import sys
# import os # Import os to construct the requirements path

# # --- Installation Step ---
# # NOTE: This is generally not recommended. Dependencies should ideally be
# # installed during the Docker image build process (in the Dockerfile)
# # for efficiency and reliability. Running this on every start adds overhead
# # and can fail if there are network issues.

# print("Attempting to install dependencies from requirements.txt...")
# # Assuming requirements.txt is in the same directory as main.py (/app in the container)
# requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')

# # Check if requirements.txt exists
# if not os.path.exists(requirements_path):
#     print(f"ERROR: requirements.txt not found at {requirements_path}")
#     sys.exit(1)

# # Construct the command
# # Using sys.executable ensures we use the same python interpreter's pip
# pip_command = [sys.executable, '-m', 'pip', 'install', '-r', requirements_path]

# # Run the command
# install_process = subprocess.run(pip_command, capture_output=True, text=True)

# # Check for errors
# if install_process.returncode != 0:
#     print("ERROR: Failed to install dependencies from requirements.txt.")
#     print("--- pip stdout ---")
#     print(install_process.stdout)
#     print("--- pip stderr ---")
#     print(install_process.stderr)
#     sys.exit(1) # Exit if installation fails
# else:
#     print("Dependencies installed successfully (or already satisfied).")
#     # Optionally print stdout for confirmation, even on success
#     # print("--- pip stdout ---")
#     # print(install_process.stdout)

# --- Original Imports (Now safe to run) ---
import argparse
from llm_tester import LLMTester
import logging

# logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)


# ---------------------------------------------------

models_to_test = [
    "deepseek-r1_8B-128k",
    "deepseek-r1_1.5B-128k",
    "qwen3_8B-128k",
    "qwen2.5_7B-128k",
    "phi3_14B_q4_medium-128k",
    "phi3_8B_q4_mini-128k",
    "llama3.1_8B-128k",
    "llama3.2_3B-128k",
    "llama3.2_1B-128k"
]

documents_to_test = [
    {
        "url": "https://www.kvgportal.com/W_global/Media/lexcom/VN/A14870/A148703540-2.pdf",
        "local_filename": "english_manual",
        "language": "english",
    },
    {
        "url": "https://www.kvgportal.com/W_global/Media/lexcom/VN/A14870/A148703640-2.pdf",
        "local_filename": "french_manual",
        "language": "french",
    },
    {
        "url": "https://www.kvgportal.com/W_global/Media/lexcom/VN/A14880/A148818240-1.pdf",
        "local_filename": "german_manual",
        "language": "german",
    },
]

# Gemma 2 is evaluator

# models_to_test = [
#     "gemini-pro",
# ]


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM tests and/or evaluations with various configurations."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        # default=["llama3.2_1B-128k"],
        default=models_to_test,
        help="List of LLM model names. Defaults to a predefined list.",
    )
    parser.add_argument(
        "--llm_type",
        type=str,
        # default="ollama",
        default="gemini",
        choices=["ollama", "gemini"],
        help="Type of LLM (ollama or gemini).",
    )
    parser.add_argument(
        "--context_type",
        type=str,
        # default="page",
        default="token",
        choices=["page", "token"],
        help="Context type (page or token).",
    )
    parser.add_argument(
        "--noise_levels",
        type=int,
        nargs="+",
        # default=[59000],
        default=[1000, 2000, 5000, 10000, 20000, 30000, 59000],
        # default=[30000],
        # default=[1000],
        # pages
        # default=[0, 10],
        # default=[0],
        help="List of noise levels (pages or tokens).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        # default="all",
        default="evaluate",
        choices=["test", "evaluate", "all"], # Added "all" choice
        help="Mode to run in: 'test' (only tests), 'evaluate' (only evaluations), or 'all' (tests then evaluations). Defaults to 'test'.",
    )

    args = parser.parse_args()

    tester = LLMTester(args.config)

    for model_name in args.models: # Outer loop: Models
        print(f"Starting processing for model: {model_name}")
        current_llm_type = "gemini" if model_name == "gemini-pro" else args.llm_type # Determine LLM type outside document loop

        if args.mode == "test" or args.mode == "all":
            print(f"\n--- Running Tests for Model: {model_name} ---")
            tester.run_tests(
                model_name,
                current_llm_type,
                args.context_type,
                args.noise_levels,
                documents_to_test, # Pass documents_to_test list
            )
            # Removed print "Finished testing" here, it's part of the overall model completion message.

        if args.mode == "evaluate" or args.mode == "all":
            # If mode is 'all', this runs after tests for the current model.
            # If mode is 'evaluate', this runs directly.
            print(f"\n--- Running Evaluations for Model: {model_name} ---")
            tester.run_evaluations(
                model_name, # Evaluate only current model
                tester.file_extensions_to_test, # Use tester's configured extensions
                args.noise_levels,
                args.context_type,
                documents_to_test, # Pass documents_to_test
            )
        
        print(f"\n--- Completed processing for Model: {model_name} (Mode: {args.mode}) ---")


if __name__ == "__main__":
    main()
# python -m utils.results_to_markdown
