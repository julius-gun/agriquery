
import os
import json
import re
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
import sys

# Add project root to path to allow importing ConfigLoader
current_script_dir = os.path.dirname(__file__)
visualization_dir = current_script_dir # Assuming this script is directly in visualization
project_root_dir = os.path.dirname(visualization_dir) # Get the parent dir (RAG)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

try:
    from utils.config_loader import ConfigLoader
    from utils.result_manager import ResultManager # Import to use sanitization logic if needed
except ImportError:
    print("Error: Could not import ConfigLoader or ResultManager. Make sure they are accessible.")
    ConfigLoader = None

# Defaults in case config is missing
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
            # Also try to infer languages from files_to_test if language_configs is missing
            loaded_langs = [lc.get("language") for lc in language_configs if lc.get("language")]
            if not loaded_langs:
                 # Fallback: check files_to_test and split by '_'
                 files = config_loader.get_files_to_test()
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
                # Sanitize extensions as they appear in filenames (replace . with _)
                sanitized_exts = [ext.replace('.', '_') for ext in extensions]
                data['extensions'] = sanitized_exts

            print(f"Loaded config data - Languages: {len(data['languages'])}, Basenames: {len(data['basenames'])}, Extensions: {len(data['extensions'])}")
            
        except Exception as e:
            print(f"Error loading data from config: {e}. Using defaults.")
            
    return data


def extract_detailed_visualization_data(results_dir: str) -> Optional[pd.DataFrame]:
    """
    Scans a directory for RAG and ZeroShot result JSON files, parses filenames
    using configuration-aware regex, and extracts metrics into a DataFrame.
    """
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found at '{results_dir}'")
        return None

    # Load config data to build robust regex
    config_data = get_config_data()
    
    # Sort by length descending to ensure regex matches longest options first (e.g. 'english_manual' before 'english')
    languages = sorted(config_data['languages'], key=len, reverse=True)
    basenames = sorted(config_data['basenames'], key=len, reverse=True)
    extensions = sorted(config_data['extensions'], key=len, reverse=True)

    # Escape for regex
    lang_pattern = '|'.join(map(re.escape, languages))
    basename_pattern = '|'.join(map(re.escape, basenames))
    ext_pattern = '|'.join(map(re.escape, extensions))

    # --- Define Regex Patterns ---
    
    # RAG Pattern (Updated for new format)
    # Format: {algo}_{basename}_{ext}_{model}_{chunk}_overlap_{overlap}_topk_{topk}.json
    # We use the specific lists of basenames and extensions to reliably split the string.
    # If basenames are empty (config fail), we might fail to match, so we include a fallback group logic or ensure defaults exist.
    rag_pattern_str = (
        rf"^(?P<algo>[^_]+)_"               # Algo (hybrid, embedding, etc) - assumes no underscores
        rf"(?P<basename>{basename_pattern})_" # File Basename (e.g., dutch_manual)
        rf"(?P<ext>{ext_pattern})_"         # Extension (e.g., md, xml)
        rf"(?P<model>.+?)_"                 # Model Name (non-greedy, captures rest)
        rf"(?P<chunk>\d+)_overlap_"         # Chunk Size
        rf"(?P<overlap>\d+)_topk_"          # Overlap Size
        rf"(?P<topk>\d+)\.json$"            # TopK
    )
    rag_pattern = re.compile(rag_pattern_str)
    print(f"Using RAG Regex with {len(basenames)} known files and {len(extensions)} extensions.")

    # ZeroShot Pattern (Kept mostly similar, but using updated languages)
    # Assumes format: zeroshot_{lang}_{model}_{ext}_{context}_{noise}_results.json
    zeroshot_pattern_str = (
        rf"^zeroshot_"
        rf"(?P<lang>{lang_pattern})_"
        rf"(?P<model>.+?)_"
        rf"(?P<ext>[a-zA-Z0-9]+)_"
        rf"(?P<context>\w+)_"
        rf"(?P<noise>\d+)_results\.json$"
    )
    zeroshot_pattern = re.compile(zeroshot_pattern_str)
    
    extracted_data = []
    skipped_files = []

    print(f"\nScanning directory: {results_dir}")

    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue 

        filepath = os.path.join(results_dir, filename)
        params = None
        
        # --- Match RAG ---
        rag_match = rag_pattern.match(filename)
        if rag_match:
            try:
                # Infer language from basename (assuming basename starts with language)
                matched_basename = rag_match.group('basename')
                inferred_lang = "unknown"
                for lang in languages:
                    if matched_basename.startswith(lang):
                        inferred_lang = lang
                        break
                
                params = {
                    'filename': filename,
                    'file_type': "RAG",
                    'retrieval_algorithm': rag_match.group('algo'),
                    'language': inferred_lang, 
                    'file_basename': matched_basename,
                    'file_extension': rag_match.group('ext'),
                    'question_model': rag_match.group('model'),
                    'chunk_size': int(rag_match.group('chunk')),
                    'overlap_size': int(rag_match.group('overlap')),
                    'num_retrieved_docs': int(rag_match.group('topk')),
                    # ZeroShot specific
                    'context_type': None,
                    'noise_level': None,
                }
                print(f"  Parsed RAG: {filename}")
            except Exception as e:
                print(f"  Error parsing RAG filename {filename}: {e}")
                skipped_files.append(filename)
                continue
        
        # --- Match ZeroShot ---
        else:
            zeroshot_match = zeroshot_pattern.match(filename)
            if zeroshot_match:
                try:
                    params = {
                        'filename': filename,
                        'file_type': "ZeroShot",
                        'retrieval_algorithm': 'zeroshot',
                        'language': zeroshot_match.group('lang'),
                        'file_basename': None, # Concept doesn't perfectly map, or is implicit
                        'file_extension': zeroshot_match.group('ext'),
                        'question_model': zeroshot_match.group('model'),
                        'context_type': zeroshot_match.group('context'),
                        'noise_level': int(zeroshot_match.group('noise')),
                        # RAG specific
                        'chunk_size': None,
                        'overlap_size': None,
                        'num_retrieved_docs': None,
                    }
                    print(f"  Parsed ZeroShot: {filename}")
                except Exception as e:
                    print(f"  Error parsing ZeroShot filename {filename}: {e}")
                    skipped_files.append(filename)
                    continue

        if params:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                overall_metrics = data.get('overall_metrics', {})
                
                # 1. Extract Overall Metrics
                for key in ['f1_score', 'accuracy', 'precision', 'recall', 'specificity']:
                    val = overall_metrics.get(key)
                    if val is not None:
                        record = params.copy()
                        record.update({'metric_type': key, 'metric_value': float(val), 'dataset_type': None})
                        extracted_data.append(record)

                # 2. Extract Dataset Success Rates
                dataset_success = overall_metrics.get('dataset_self_evaluation_success', {})
                if isinstance(dataset_success, dict):
                    for ds_name, rate in dataset_success.items():
                        if rate is not None:
                            record = params.copy()
                            record.update({'metric_type': 'dataset_success', 'metric_value': float(rate), 'dataset_type': ds_name})
                            extracted_data.append(record)

            except json.JSONDecodeError:
                print(f"Warning: JSON decode error in {filename}. Skipping.")
                skipped_files.append(filename)
            except Exception as e:
                print(f"Warning: Unexpected error processing {filename}: {e}")
                skipped_files.append(filename)
        else:
            if filename.endswith(".json"):
                # print(f"  Skipping unmatched file: {filename}")
                skipped_files.append(filename + " (no regex match)")

    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} files (errors or no match).")

    if not extracted_data:
        print("No valid result data extracted.")
        return None

    df = pd.DataFrame(extracted_data)
    
    # Organize columns
    cols = [
        'filename', 'file_type', 'retrieval_algorithm', 'language', 'file_basename', 
        'file_extension', 'question_model', 'chunk_size', 'overlap_size', 
        'num_retrieved_docs', 'context_type', 'noise_level', 
        'metric_type', 'metric_value', 'dataset_type'
    ]
    # Filter to cols that exist
    final_cols = [c for c in cols if c in df.columns]
    df = df[final_cols]

    print(f"\nSuccessfully extracted {len(extracted_data)} rows.")
    return df

if __name__ == '__main__':
    default_results_dir = os.path.join(project_root_dir, 'results')
    print(f"\n--- Testing Detailed Data Extractor ---")
    df_results = extract_detailed_visualization_data(default_results_dir)
    if df_results is not None:
        print(df_results.head())
        print("\nColumn Types:")
        print(df_results.dtypes)
