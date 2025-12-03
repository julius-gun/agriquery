import seaborn as sns

# --- Global Style Settings ---
sns.set_theme(style="whitegrid")

# --- Language Configuration ---
# Consistent colors for all plots
LANGUAGE_PALETTE = {
    "english": "#1f77b4",  # Blue
    "german": "#2ca02c",   # Green
    "french": "#ff7f0e",   # Orange
    "dutch": "#9467bd",    # Purple
    "italian": "#d62728",  # Red
    "spanish": "#8c564b"   # Brown
}

LANGUAGE_ORDER = [
    "english", 
    "german", 
    "french", 
    "dutch", 
    "italian", 
    "spanish"
]

LANGUAGE_CODES = {
    "english": "EN",
    "german": "DE",
    "french": "FR",
    "dutch": "NL",
    "italian": "IT",
    "spanish": "ES"
}

# --- Format Configuration ---
FORMAT_PALETTE = {
    "Markdown": "#333333", # Dark Grey
    "JSON": "#e377c2",     # Pink
    "XML": "#7f7f7f"       # Grey
}

FORMAT_ORDER = ["Markdown", "JSON", "XML"]

FORMAT_CODES = {
    "Markdown": "MD",
    "JSON": "JSON",
    "XML": "XML"
}

# --- Metric Display Names ---
METRIC_DISPLAY_NAMES = {
    "f1_score": "F1 Score",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
}

# --- Model Name Cleaning ---
MODEL_NAME_MAPPINGS = {
    # Existing / Other
    "gemini-2.5-flash-preview-04-17": "Gemini 2.5 Flash",
    "phi3_14B_q4_medium-128k": "Phi3 14B",

    # DeepSeek
    "deepseek-r1_8B-128k": "DeepSeek R1 8B",
    "deepseek-r1_1.5B-128k": "DeepSeek R1 1.5B",

    # Qwen
    "qwen2.5_7B-128k": "Qwen 2.5 7B",
    "qwen3_8B-128k": "Qwen 3 8B",

    # Llama
    "llama3.1_8B-128k": "Llama 3.1 8B",
    "llama3.2_3B-128k": "Llama 3.2 3B",

    # Gemma
    "gemma3_12B-128k": "Gemma 3 12B",

    # GPT-OSS
    "gpt-oss-20b_128k": "GPT-OSS 20B",
    "gpt-oss-120b_128k": "GPT-OSS 120B",
    "gpt-oss-120b-cloud_128k": "GPT-OSS 120B Cloud"
}

# --- Model Parameters (Billions) ---
# Used for size vs performance scatter plots
# Keys should match the values in MODEL_NAME_MAPPINGS (Cleaned Names)
MODEL_PARAMS = {
    "DeepSeek R1 1.5B": 1.5,
    "Llama 3.2 3B": 3.0,
    "Qwen 2.5 7B": 7.0,
    "DeepSeek R1 8B": 8.0,
    "Llama 3.1 8B": 8.0,
    "Qwen 3 8B": 8.0,
    "Gemma 3 12B": 12.0,
    "Phi3 14B": 14.0,
    "GPT-OSS 20B": 20.0,
    "GPT-OSS 120B": 120.0,
    "GPT-OSS 120B Cloud": 120.0
}

def clean_model_name(model_name: str) -> str:
    """Standardizes model names for plots."""
    name = MODEL_NAME_MAPPINGS.get(model_name, model_name)
    
    # Generic cleanup if not explicitly mapped
    if name == model_name:
        name = name.replace("-128k", "").replace("_", " ")
        if "preview" in name:
            name = name.split("-preview")[0]
            
    return name
