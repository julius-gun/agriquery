import os
import sys
import pathlib
from typing import List, Dict, Any

# --- Path Setup ---
# Add the project root (RAG2_COMPAG) to the Python path to allow importing modules
project_root = pathlib.Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.config_loader import ConfigLoader
    from retrieval_pipelines.embedding_retriever import EmbeddingRetriever
except ImportError as e:
    print(f"Error: Failed to import necessary modules. Ensure the script is run from a correct location.")
    print(f"Project Root: {project_root}")
    print(f"Details: {e}")
    sys.exit(1)


class TokenCounter:
    """
    A utility to count characters and tokens for specified manuals.
    """

    def __init__(self, config_path: str = "config.json"):
        """Initializes the TokenCounter by loading configuration and the tokenizer."""
        print("--- Initializing Token Counter ---")
        absolute_config_path = project_root / config_path
        self.config_loader = ConfigLoader(str(absolute_config_path))
        
        self.manuals_dir = project_root / "manuals"
        self.files_to_test = self.config_loader.get_files_to_test()
        self.extensions_to_test = self.config_loader.get_file_extensions_to_test()
        
        print("Loading tokenizer from EmbeddingRetriever...")
        # We use EmbeddingRetriever as it contains the tokenizer used for chunking.
        # This gives a consistent token count relative to the RAG process.
        embedding_model_config = self.config_loader.get_embedding_model_config()
        retriever = EmbeddingRetriever(model_config=embedding_model_config)
        self.tokenizer = retriever.tokenizer
        print("Tokenizer loaded successfully.")

    def count_tokens_for_all_manuals(self) -> List[Dict[str, Any]]:
        """
        Iterates through all configured manuals, counts characters and tokens.

        Returns:
            A list of dictionaries, each containing statistics for one manual file.
        """
        results = []
        print(f"\n--- Counting Tokens for Manuals in '{self.manuals_dir}' ---")

        for file_basename in self.files_to_test:
            for extension in self.extensions_to_test:
                filename = f"{file_basename}.{extension}"
                filepath = self.manuals_dir / filename

                if not filepath.is_file():
                    print(f"Skipping, file not found: {filepath}")
                    continue

                print(f"Processing: {filename}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                    char_count = len(content)
                    
                    # Tokenize the content and get the number of tokens
                    # The tokenizer returns a dictionary with 'input_ids', the length of which is the token count.
                    token_ids = self.tokenizer(content, return_tensors=None, add_special_tokens=False)['input_ids']
                    token_count = len(token_ids)

                    results.append({
                        "file_basename": file_basename,
                        "format": extension,
                        "characters": char_count,
                        "tokens": token_count
                    })

                except Exception as e:
                    print(f"  Error processing file {filename}: {e}")
                    results.append({
                        "file_basename": file_basename,
                        "format": extension,
                        "error": str(e)
                    })
        
        return results

    @staticmethod
    def print_results_table(results: List[Dict[str, Any]]):
        """
        Prints the token count results in a formatted Markdown table.
        """
        if not results:
            print("No results to display.")
            return
            
        print("\n--- Token Count Summary ---")
        
        # Sort results for consistent output
        # Sort by file_basename, then by a predefined order of formats
        format_order = {'md': 0, 'json': 1, 'xml': 2}
        results.sort(key=lambda x: (x['file_basename'], format_order.get(x['format'], 99)))

        # Print Markdown table header
        print("\n| Language (File)  | Format   | Characters | Tokens     |")
        print("|------------------|----------|------------|------------|")

        for res in results:
            if "error" in res:
                print(f"| {res['file_basename']:<16} | {res['format']:<8} | Error processing file |")
            else:
                print(f"| {res['file_basename']:<16} | {res['format']:<8} | {res['characters']:<10,} | {res['tokens']:<10,} |")
        
        print("\nTable is in Markdown format for easy copy-pasting into your paper.")


def main():
    """Main function to run the token counter and print the results."""
    try:
        counter = TokenCounter()
        results = counter.count_tokens_for_all_manuals()
        counter.print_results_table(results)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Token Counter Script Finished ---")


if __name__ == "__main__":
    main()
