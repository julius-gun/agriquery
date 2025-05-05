import argparse
from llm_tester import LLMTester
import logging

# logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)


# ---------------------------------------------------

models_to_test = [
    "llama3.2_1B-128k",
    "deepseek-r1_1.5B-128k",
    "deepseek-r1_8B-128k",
    "qwen3_8B-128k",
    "qwen2.5_7B-128k",
    "phi3_14B_q4_medium-128k",
    "phi3_8B_q4_mini-128k",
    "llama3.1_8B-128k",
    "llama3.2_3B-128k",
    "phi3_14B_medium-4k"
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
        default="ollama",
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
        default="test",
        # default="evaluate",
        choices=["test", "evaluate"],
        help="Mode to run in: 'test' or 'evaluate'. Defaults to 'test'.",
    )

    args = parser.parse_args()

    tester = LLMTester(args.config)

    for model_name in args.models: # Outer loop: Models
        print(f"Starting processing for model: {model_name}")
        current_llm_type = "gemini" if model_name == "gemini-pro" else args.llm_type # Determine LLM type outside document loop

        if args.mode == "test":
            tester.run_tests(
                model_name,
                current_llm_type,
                args.context_type,
                args.noise_levels,
                documents_to_test, # Pass documents_to_test list
            )
            args.mode == "evaluate" # Evaluate after testing
            print(f"Finished testing {model_name}")
        if args.mode == "evaluate":
             tester.run_evaluations(
                model_name, # Evaluate only current model
                tester.file_extensions_to_test,
                args.noise_levels,
                args.context_type,
                documents_to_test, # Pass documents_to_test
            )
             print(f"Finished evaluation for {model_name}")


if __name__ == "__main__":
    main()
# python -m utils.results_to_markdown
