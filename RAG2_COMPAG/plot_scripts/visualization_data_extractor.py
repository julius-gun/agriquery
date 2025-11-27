import os
import json
import re
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
import sys

# Add project root to path to allow importing ConfigLoader
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming this script is in a subdirectory like 'visualization' or 'plot_scripts'
# and the project root is its parent directory.
project_root_dir = os.path.dirname(current_script_dir) 
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

try:
    from utils.config_loader import ConfigLoader
    from utils.result_manager import ResultManager # Imported for potential future use or context
except ImportError:
    print("Error: Could not import ConfigLoader or ResultManager. Make sure they are accessible.")
    ConfigLoader = None

# Defaults in case config is missing or inaccessible
DEFAULT_LANGUAGES = ['english', 'french', 'german', 'dutch', 'spanish', 'italian']
DEFAULT_EXTENSIONS = ['md', 'json', 'xml', 'txt', 'html']
# Default basenames might be hard to guess, but we can try generic ones if config fails
DEFAULT_BASENAMES = [f"{lang}_manual" for lang in DEFAULT_LANGUAGES]

def get_config_data(config_path: str = 'config.json') -> Dict[str, List[str]]:
    """
    Loads configuration data relevant for parsing filenames:
    languages, files_to_test (basenames), and file_extensions.
    """
    data = {
        'languages': DEFAULT_LANGUAGES,
        'basenames': DEFAULT_BASENAMES,
        'extensions': DEFAULT_EXTENSIONS
    }
    
    if ConfigLoader:
        try:
            abs_config_path = os.path.join(project_root_dir, config_path)
            if not os.path.exists(abs_config_path):
                 print(f"Warning: Config file not found at '{abs_config_path}'. Using defaults.")
                 return data

            config_loader = ConfigLoader(abs_config_path)
            
            # 1. Languages
            language_configs = config_loader.config.get("language_configs", [])
            loaded_langs = [lc.get("language") for lc in language_configs if lc.get("language")]
            
            # Fallback: try to infer languages from files_to_test if language_configs is missing
            if not loaded_langs:
                 files = config_loader.get_files_to_test()
                 # Assumes files_to_test are typically in a 'lang_something' format
                 loaded_langs = list(set([f.split('_')[0] for f in files if '_' in f]))
            
            if loaded_langs:
                data['languages'] = loaded_langs

            # 2. Basenames (files_to_test)
            files_to_test = config_loader.get_files_to_test()
            if files_to_test:
                data['basenames'] = files_to_test
            
            # 3. Extensions
            extensions = config_loader.get_file_extensions_to_test()
            if extensions:
                # Sanitize extensions for regex matching in filenames (replace . with _)
                sanitized_exts = [ext.replace('.', '_') for ext in extensions]
                data['extensions'] = sanitized_exts

            print(f"Loaded config data - Languages: {len(data['languages'])}, Basenames: {len(data['basenames'])}, Extensions: {len(data['extensions'])}")
            
        except Exception as e:
            print(f"Error loading data from config: {e}. Using defaults.")
            
    return data

def _build_regex_pattern_group(items: List[str]) -> str:
    """
    Builds a regex pattern string for a group of items, escaping them and
    joining with '|' for 'OR' matching. Sorts by length descending for greedy matching.
    """
    # Sort by length descending to ensure regex matches longest options first
    # (e.g., 'english_manual' before 'english')
    sorted_items = sorted(items, key=len, reverse=True)
    return '|'.join(map(re.escape, sorted_items))

def _parse_rag_filename(filename: str, rag_pattern: re.Pattern, languages: List[str]) -> Optional[Dict[str, Any]]:
    """
    Parses a RAG filename using the compiled regex pattern and extracts parameters.
    Infers language from the matched basename.
    """
    match = rag_pattern.match(filename)
    if not match:
        return None
    
    try:
        matched_basename = match.group('basename')
        inferred_lang = "unknown"
        # Infer language from basename (assuming basename starts with language)
        for lang in languages:
            if matched_basename.startswith(lang):
                inferred_lang = lang
                break
        
        return {
            'filename': filename,
            'file_type': "RAG",
            'retrieval_algorithm': match.group('algo'),
            'language': inferred_lang, 
            'file_basename': matched_basename,
            'file_extension': match.group('ext'),
            'question_model': match.group('model'),
            'chunk_size': int(match.group('chunk')),
            'overlap_size': int(match.group('overlap')),
            'num_retrieved_docs': int(match.group('topk')),
            # ZeroShot specific fields are None for RAG
            'context_type': None,
            'noise_level': None,
        }
    except Exception as e:
        print(f"  Error parsing RAG filename {filename}: {e}")
        return None

def _parse_zeroshot_filename(filename: str, zeroshot_pattern: re.Pattern) -> Optional[Dict[str, Any]]:
    """
    Parses a ZeroShot filename using the compiled regex pattern and extracts parameters.
    """
    match = zeroshot_pattern.match(filename)
    if not match:
        return None
    
    try:
        return {
            'filename': filename,
            'file_type': "ZeroShot",
            'retrieval_algorithm': 'zeroshot',
            'language': match.group('lang'),
            'file_basename': None, # Concept doesn't perfectly map for ZeroShot
            'file_extension': match.group('ext'),
            'question_model': match.group('model'),
            'context_type': match.group('context'),
            'noise_level': int(match.group('noise')),
            # RAG specific fields are None for ZeroShot
            'chunk_size': None,
            'overlap_size': None,
            'num_retrieved_docs': None,
        }
    except Exception as e:
        print(f"  Error parsing ZeroShot filename {filename}: {e}")
        return None

def _extract_metrics_from_file(filepath: str, base_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Loads a JSON result file and extracts overall metrics and dataset success rates.
    Returns a list of dictionaries, each representing a single metric record.
    """
    extracted_records = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        overall_metrics = data.get('overall_metrics', {})
        
        # 1. Extract standard overall metrics
        for key in ['f1_score', 'accuracy', 'precision', 'recall', 'specificity']:
            val = overall_metrics.get(key)
            if val is not None:
                record = base_params.copy()
                record.update({'metric_type': key, 'metric_value': float(val), 'dataset_type': None})
                extracted_records.append(record)

        # 2. Extract dataset self-evaluation success rates
        dataset_success = overall_metrics.get('dataset_self_evaluation_success', {})
        if isinstance(dataset_success, dict):
            for ds_name, rate in dataset_success.items():
                if rate is not None:
                    record = base_params.copy()
                    record.update({'metric_type': 'dataset_success', 'metric_value': float(rate), 'dataset_type': ds_name})
                    extracted_records.append(record)

    except json.JSONDecodeError:
        print(f"Warning: JSON decode error in {os.path.basename(filepath)}. Skipping metrics.")
    except Exception as e:
        print(f"Warning: Unexpected error processing metrics from {os.path.basename(filepath)}: {e}")
            
    return extracted_records


def extract_detailed_visualization_data(results_dir: str) -> Optional[pd.DataFrame]:
    """
    Scans a directory for RAG and ZeroShot result JSON files, parses filenames
    using configuration-aware regex, and extracts metrics into a DataFrame.
    """
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found at '{results_dir}'")
        return None

    # Load config data to build robust regex patterns
    config_data = get_config_data()
    
    # Build regex patterns for dynamic parts of filenames
    lang_pattern_str = _build_regex_pattern_group(config_data['languages'])
    basename_pattern_str = _build_regex_pattern_group(config_data['basenames'])
    ext_pattern_str = _build_regex_pattern_group(config_data['extensions'])

    # --- Define and Compile Regex Patterns ---
    
    # RAG Pattern: {algo}_{basename}_{ext}_{model}_{chunk}_overlap_{overlap}_topk_{topk}.json
    rag_pattern_str = (
        rf"^(?P<algo>[^_]+)_"               # Retrieval Algorithm (e.g., hybrid, embedding)
        rf"(?P<basename>{basename_pattern_str})_" # File Basename (e.g., dutch_manual)
        rf"(?P<ext>{ext_pattern_str})_"     # File Extension (e.g., md, xml)
        rf"(?P<model>.+?)_"                 # Question Model Name (non-greedy)
        rf"(?P<chunk>\d+)_overlap_"         # Chunk Size
        rf"(?P<overlap>\d+)_topk_"          # Overlap Size
        rf"(?P<topk>\d+)\.json$"            # TopK (number of retrieved documents)
    )
    rag_pattern = re.compile(rag_pattern_str)
    print(f"Using RAG Regex with {len(config_data['basenames'])} known files and {len(config_data['extensions'])} extensions.")

    # ZeroShot Pattern: zeroshot_{lang}_{model}_{ext}_{context}_{noise}_results.json
    zeroshot_pattern_str = (
        rf"^zeroshot_"
        rf"(?P<lang>{lang_pattern_str})_"
        rf"(?P<model>.+?)_"
        rf"(?P<ext>[a-zA-Z0-9]+)_"          # Allows any alphanumeric for extension in ZeroShot (could use ext_pattern_str too)
        rf"(?P<context>\w+)_"               # Context type (e.g., full, summary)
        rf"(?P<noise>\d+)_results\.json$"   # Noise level
    )
    zeroshot_pattern = re.compile(zeroshot_pattern_str)
    
    extracted_data = []
    skipped_files = []

    print(f"\nScanning directory: {results_dir}")

    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue 

        filepath = os.path.join(results_dir, filename)
        base_params = None # Parameters extracted from filename, before metrics

        # Attempt to parse as RAG file
        base_params = _parse_rag_filename(filename, rag_pattern, config_data['languages'])
        if base_params:
            print(f"  Parsed RAG: {filename}")
        else:
            # If not RAG, attempt to parse as ZeroShot file
            base_params = _parse_zeroshot_filename(filename, zeroshot_pattern)
            if base_params:
                print(f"  Parsed ZeroShot: {filename}")
        
        if base_params:
            # If filename was successfully parsed, extract metrics from the JSON content
            file_records = _extract_metrics_from_file(filepath, base_params)
            extracted_data.extend(file_records)
        else:
            skipped_files.append(filename + " (no regex match)")

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files (errors or no match).")

    if not extracted_data:
        print("No valid result data extracted.")
        return None

    df = pd.DataFrame(extracted_data)
    
    # Organize columns for better readability and consistency
    cols = [
        'filename', 'file_type', 'retrieval_algorithm', 'language', 'file_basename', 
        'file_extension', 'question_model', 'chunk_size', 'overlap_size', 
        'num_retrieved_docs', 'context_type', 'noise_level', 
        'metric_type', 'metric_value', 'dataset_type'
    ]
    # Filter to include only columns that actually exist in the DataFrame
    final_cols = [c for c in cols if c in df.columns]
    df = df[final_cols]

    print(f"\nSuccessfully extracted {len(extracted_data)} rows.")
    return df

if __name__ == '__main__':
    # Determine the default results directory relative to the project root
    default_results_dir = os.path.join(project_root_dir, 'results')
    print(f"\n--- Testing Detailed Data Extractor ---")
    
    # Example usage: extract data from the default results directory
    df_results = extract_detailed_visualization_data(default_results_dir)
    
    if df_results is not None:
        print("\n--- Extracted Data Head ---")
        print(df_results.head())
        print("\n--- Column Types ---")
        print(df_results.dtypes)
        print(f"\nTotal rows extracted: {len(df_results)}")
    else:
        print("No DataFrame was generated.")
