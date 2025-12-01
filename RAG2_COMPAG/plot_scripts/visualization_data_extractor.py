import os
import json
import re
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
import sys

# Add project root path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_script_dir) 
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

try:
    from utils.config_loader import ConfigLoader
except ImportError:
    ConfigLoader = None

# Defaults
DEFAULT_LANGUAGES = ['english', 'french', 'german', 'dutch', 'spanish', 'italian']
DEFAULT_EXTENSIONS = ['md', 'json', 'xml']
DEFAULT_BASENAMES = [f"{lang}_manual" for lang in DEFAULT_LANGUAGES]

def get_config_data(config_path: str = 'config.json') -> Dict[str, List[str]]:
    """Loads configuration data for regex building."""
    data = {
        'languages': DEFAULT_LANGUAGES,
        'basenames': DEFAULT_BASENAMES,
        'extensions': DEFAULT_EXTENSIONS
    }
    
    if ConfigLoader:
        try:
            abs_config_path = os.path.join(project_root_dir, config_path)
            if os.path.exists(abs_config_path):
                config_loader = ConfigLoader(abs_config_path)
                
                # Languages
                lc = config_loader.config.get("language_configs", [])
                loaded = [x.get("language") for x in lc if x.get("language")]
                if loaded: data['languages'] = loaded

                # Files
                ft = config_loader.get_files_to_test()
                if ft: data['basenames'] = ft
                
                # Extensions
                exts = config_loader.get_file_extensions_to_test()
                if exts: data['extensions'] = [e.replace('.', '_') for e in exts]
        except Exception as e:
            print(f"Warning loading config: {e}")
            
    return data

def _build_regex_pattern(items: List[str]) -> str:
    # Sort by length descending for greedy regex matching
    return '|'.join(map(re.escape, sorted(items, key=len, reverse=True)))

def _parse_filename(filename: str, pattern: re.Pattern, languages: List[str]) -> Optional[Dict[str, Any]]:
    """Parses RAG filename."""
    match = pattern.match(filename)
    if not match:
        return None
    
    try:
        basename = match.group('basename')
        lang = "unknown"
        for l in languages:
            if basename.startswith(l):
                lang = l
                break
        
        return {
            'filename': filename,
            'retrieval_algorithm': match.group('algo'),
            'language': lang,
            'file_basename': basename,
            'file_extension': match.group('ext'),
            'question_model': match.group('model'),
            'chunk_size': int(match.group('chunk')),
            'overlap_size': int(match.group('overlap')),
            'num_retrieved_docs': int(match.group('topk'))
        }
    except Exception:
        return None

def extract_detailed_visualization_data(results_dir: str) -> Optional[pd.DataFrame]:
    """
    Extracts metrics from RAG result files.
    Note: ZeroShot and Noise Level logic has been removed as per requirements.
    """
    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found {results_dir}")
        return None

    config = get_config_data()
    
    # RAG Pattern: {algo}_{basename}_{ext}_{model}_{chunk}_overlap_{overlap}_topk_{topk}.json
    pattern_str = (
        rf"^(?P<algo>[^_]+)_"
        rf"(?P<basename>{_build_regex_pattern(config['basenames'])})_"
        rf"(?P<ext>{_build_regex_pattern(config['extensions'])})_"
        rf"(?P<model>.+?)_"
        rf"(?P<chunk>\d+)_overlap_"
        rf"(?P<overlap>\d+)_topk_"
        rf"(?P<topk>\d+)\.json$"
    )
    rag_pattern = re.compile(pattern_str)
    
    extracted = []

    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"): continue
        
        # Skip if it looks like a zeroshot file (optimization)
        if filename.startswith("zeroshot_"): continue

        params = _parse_filename(filename, rag_pattern, config['languages'])
        if not params: continue
        
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract Metrics
            overall = data.get('overall_metrics', {})
            for m in ['f1_score', 'accuracy', 'precision', 'recall']:
                val = overall.get(m)
                if val is not None:
                    rec = params.copy()
                    rec.update({'metric_type': m, 'metric_value': float(val)})
                    extracted.append(rec)
                    
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not extracted:
        return None

    return pd.DataFrame(extracted)
