# RAG2_COMPAG/analysis/token_counter_individual_tokenizers.py
import sys
import pathlib
import os
from typing import List, Dict, Any, Tuple

# --- Path Setup ---
# Add the project root (RAG2_COMPAG) to the Python path to allow importing modules
project_root = pathlib.Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Dependency Imports ---
try:
    from transformers import AutoTokenizer
    import tiktoken
except ImportError as e:
    print("Error: Missing required libraries. Please install them:")
    print("pip install transformers tiktoken")
    print(f"Details: {e}")
    sys.exit(1)

try:
    from utils.config_loader import ConfigLoader
except ImportError as e:
    print(f"Error: Failed to import ConfigLoader. Ensure the script is run from a correct location.")
    print(f"Project Root: {project_root}")
    print(f"Details: {e}")
    sys.exit(1)


class TokenCounter:
    """
    A utility to count characters and tokens for specified manuals using multiple tokenizers.
    """

    def __init__(self, config_path: str = "config.json"):
        """Initializes the TokenCounter by loading configuration."""
        print("--- Initializing Token Counter ---")
        absolute_config_path = project_root / config_path
        self.config_loader = ConfigLoader(str(absolute_config_path))
        
        self.manuals_dir = project_root / "manuals"
        self.files_to_test = self.config_loader.get_files_to_test()
        self.extensions_to_test = self.config_loader.get_file_extensions_to_test()
        
        self.tokenizers_to_analyze = self.config_loader.config.get("tokenizers_for_analysis", [])
        if not self.tokenizers_to_analyze:
            print("Error: 'tokenizers_for_analysis' list not found or is empty in config.json.")
            sys.exit(1)
        
        self.loaded_tokenizers = self._load_all_tokenizers()

    def _load_tokenizer(self, tokenizer_config: Dict[str, str]) -> Any:
        """Loads a single tokenizer based on its configuration."""
        name = tokenizer_config.get("name")
        tok_type = tokenizer_config.get("type")
        path = tokenizer_config.get("path")

        if not all([name, tok_type, path]):
            print(f"  Skipping invalid tokenizer config: {tokenizer_config}")
            return None
        
        print(f"  Loading tokenizer: '{name}' ({tok_type} from '{path}')...")
        try:
            if tok_type == "huggingface":
                hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
                if hf_token:
                    print("  Hugging Face token found in environment.")
                    return AutoTokenizer.from_pretrained(path, token=hf_token)
                else:
                    print("  WARNING: HUGGING_FACE_HUB_TOKEN environment variable not found.")
                    # Attempt to load without a token as a fallback, though likely to fail for gated models.
                    return AutoTokenizer.from_pretrained(path)
            elif tok_type == "tiktoken":
                return tiktoken.get_encoding(path)
            else:
                print(f"  Unsupported tokenizer type: '{tok_type}' for '{name}'")
                return None
        except Exception as e:
            print(f"  Failed to load tokenizer '{name}': {e}")
            return None

    def _load_all_tokenizers(self) -> Dict[str, Dict[str, Any]]:
        """Pre-loads all tokenizers specified in the configuration."""
        print("\n--- Loading All Specified Tokenizers ---")
        tokenizers = {}
        for config in self.tokenizers_to_analyze:
            tokenizer_name = config.get("name")
            tok_type = config.get("type")
            if tokenizer_name and tok_type:
                tokenizer_obj = self._load_tokenizer(config)
                if tokenizer_obj:
                    tokenizers[tokenizer_name] = {
                        "tokenizer": tokenizer_obj,
                        "type": tok_type
                    }
        print("--- Tokenizer Loading Complete ---")
        return tokenizers

    def _count_tokens_with_tokenizer(self, tokenizer_info: Dict[str, Any], content: str) -> int:
        """Counts tokens for a given content using a specific loaded tokenizer."""
        tokenizer = tokenizer_info["tokenizer"]
        tok_type = tokenizer_info["type"]
        
        try:
            if tok_type == "huggingface":
                return len(tokenizer.encode(content, add_special_tokens=False))
            elif tok_type == "tiktoken":
                return len(tokenizer.encode(content))
            else:
                return -1
        except Exception as e:
            print(f"    Error during tokenization with '{tok_type}': {e}")
            return -1

    def count_tokens_for_all_manuals(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Iterates through all configured manuals and tokenizers to count tokens.

        Returns:
            A dictionary grouping results by file.
            Example: {('english_manual', 'md'): {'characters': 123, 'tokens': {'TokenizerA': 100}}}
        """
        results = {}
        print(f"\n--- Counting Tokens for Manuals in '{self.manuals_dir}' ---")

        for file_basename in self.files_to_test:
            for extension in self.extensions_to_test:
                filename = f"{file_basename}.{extension}"
                filepath = self.manuals_dir / filename
                file_key = (file_basename, extension)

                if not filepath.is_file():
                    print(f"Skipping, file not found: {filepath}")
                    continue

                print(f"Processing: {filename}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    char_count = len(content)
                    token_counts = {}

                    for name, tokenizer_info in self.loaded_tokenizers.items():
                        count = self._count_tokens_with_tokenizer(tokenizer_info, content)
                        token_counts[name] = count

                    results[file_key] = {
                        "characters": char_count,
                        "tokens": token_counts
                    }

                except Exception as e:
                    print(f"  Error processing file {filename}: {e}")
                    results[file_key] = {"error": str(e)}
        
        return results

    def print_results_table(self, results: Dict[Tuple[str, str], Dict[str, Any]]):
        """
        Prints the token count results in a formatted Markdown table for side-by-side comparison.
        """
        if not results:
            print("No results to display.")
            return
            
        print("\n--- Token Count Comparison Summary ---")
        
        tokenizer_names = list(self.loaded_tokenizers.keys())
        
        header = "| Language (File)  | Format   | Characters "
        separator = "|------------------|----------|------------"
        
        for name in tokenizer_names:
            header += f"| {name} "
            separator += "|-" + "-" * (len(name) + 1)
        
        header += "|"
        separator += "|"
        
        print(header)
        print(separator)

        format_order = {'md': 0, 'json': 1, 'xml': 2}
        sorted_keys = sorted(results.keys(), key=lambda k: (k[0], format_order.get(k[1], 99)))

        for key in sorted_keys:
            file_basename, ext = key
            res = results[key]
            
            if "error" in res:
                row = f"| {file_basename:<16} | {ext:<8} | Error: {res['error']}"
                for _ in tokenizer_names:
                    row += "| "
                row += "|"
                print(row)
                continue

            row = f"| {file_basename:<16} | {ext:<8} | {res['characters']:<10,} "
            
            for name in tokenizer_names:
                token_count = res["tokens"].get(name, -1)
                col_width = len(name)
                if token_count != -1:
                    row += f"| {token_count:<{col_width},} "
                else:
                    row += f"| {'Error':<{col_width}} "

            row += "|"
            print(row)
        
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