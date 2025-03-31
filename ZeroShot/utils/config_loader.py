# utils/config_loader.py
import json
import os
import re


class ConfigLoader:
    """Loads configurations from a JSON file and prompt templates."""

    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(
                f"Warning: Configuration file not found at '{self.config_path}'. Using default configurations."
            )
            return self._load_default_config()
        except json.JSONDecodeError:
            raise ValueError(
                f"Error decoding JSON from '{self.config_path}'. Please check if the file is valid JSON."
            )

    def _load_default_config(self):
        default_config = {
            "llm_models": {
                "ollama": {
                    "deepseek-r1_70B-128k": {
                        "name": "deepseek-r1:70b",
                        "temperature": 0.0,
                        "num_predict": 512,
                        "context_window": 128000,
                    },
                    "deepseek-r1_1.5B-128k": {
                        "name": "deepseek-r1:1.5b",
                        "temperature": 0.0,
                        "num_predict": 512,
                        "context_window": 128000,
                    },
                    "deepseek-r1_14B-128k": {
                        "name": "deepseek-r1:14b",
                        "temperature": 0.0,
                        "num_predict": 512,
                        "context_window": 128000,
                    },
                    "qwen2.5_7B-128k": {
                        "name": "qwen2.5:latest",
                        "temperature": 0.0,
                        "num_predict": 512,
                        "context_window": 128000,
                    },
                    "phi3_14B_q4_medium-128k": {
                        "name": "phi3:medium-128k",
                        "temperature": 0.0,
                        "num_predict": 512,
                        "context_window": 128000,
                    },
                    "phi3_8B_q4_mini-128k": {
                        "name": "phi3:mini-128k",
                        "temperature": 0.0,
                        "num_predict": 512,
                        "context_window": 128000,
                    },
                    "llama3.1_8B-128k": {
                        "name": "llama3.1:latest",
                        "temperature": 0.0,
                        "num_predict": 512,
                        "context_window": 128000,
                    },
                    "llama3.2_3B-128k": {
                        "name": "llama3.2:latest",
                        "temperature": 0.0,
                        "num_predict": 512,
                        "context_window": 128000,
                    },
                    "llama3.2_1B-128k": {
                        "name": "llama3.2:1b",
                        "temperature": 0.0,
                        "num_predict": 512,
                        "context_window": 128000,
                    },
                    "phi3_14B_medium-4k": {
                        "name": "phi3:medium",
                        "temperature": 0.0,
                        "num_predict": 512,
                        "context_window": 4000,
                    },
                    "gemma2_9B-8k": {
                        "name": "gemma2:latest",
                        "temperature": 0.0,
                        "num_predict": 512,
                        "context_window": 8192,
                    },
                    "phi4_14B-4k": {
                        "name": "phi4:latest",
                        "temperature": 0.0,
                        "num_predict": 512,
                        "context_window": 4096,
                    },
                },
                "gemini": {
                    "gemini-pro": {
                        "name": "gemini-2.0-flash-thinking-exp-01-21",
                        "temperature": 0.0,
                        "num_predict": 512,
                    }
                },
            },
            "evaluator_model": "gemma2:latest",
            "prompt_paths": {
                "question_prompt": "prompt_templates/question_prompt.txt",
                "evaluation_prompt": "prompt_templates/evaluation_prompt.txt",
            },
            "question_dataset_paths": [
                "question_datasets/question_answers_pairs.json",
                "question_datasets/question_answers_tables.json",
                "question_datasets/question_answers_unanswerable.json",
            ],
            "document_path": "https://www.kvgportal.com/W_global/Media/lexcom/VN/A14870/A148703540-2.pdf",
            "output_dir": "results",
            "question_tracker_path": "utils/question_tracker_data.json",
            "tokenizer_name": "pcuenq/Llama-3.2-1B-Instruct-tokenizer",
            "file_extensions_to_test": ["txt"],
            "visualization_dir": "plots" # default value in default config, too
        }
        return default_config

    def get_llm_models_config(self, llm_type="ollama"):
        llm_type = llm_type.lower()
        if llm_type not in self.config["llm_models"]:
            raise ValueError(f"LLM type '{llm_type}' not supported in config.")
        return self.config["llm_models"][llm_type]

    def get_evaluator_model_name(self):
        return self.config["evaluator_model"]

    def get_prompt_path(self, prompt_name):
        if prompt_name not in self.config["prompt_paths"]:
            raise ValueError(f"Prompt template '{prompt_name}' not defined in config.")
        return self.config["prompt_paths"][prompt_name]

    def get_question_dataset_paths(self):
        return self.config["question_dataset_paths"]

    def get_document_path(self):
        return self.config["document_path"]

    def get_output_dir(self):
        return self.config["output_dir"]

    def get_question_tracker_path(self):
        return self.config.get("question_tracker_path", "utils/question_tracker_data.json") # default path if not set

    def get_visualization_dir(self):
        return self.config.get("visualization_dir", "visualization_plots") # default value if not in config

    def load_prompt_template(self, prompt_name):
        prompt_path = self.get_prompt_path(prompt_name)
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prompt template file not found at '{prompt_path}'."
            )
        except Exception as e:
            raise RuntimeError(
                f"Error reading prompt template from '{prompt_path}': {e}"
            )


if __name__ == "__main__":
    config_loader = ConfigLoader()

    ollama_models = config_loader.get_llm_models_config("ollama")
    print("Ollama models:", ollama_models)

    evaluator_model = config_loader.get_evaluator_model_name()
    print("Evaluator model:", evaluator_model)

    question_prompt_path = config_loader.get_prompt_path("question_prompt")
    print("Question prompt path:", question_prompt_path)

    question_prompt_content = config_loader.load_prompt_template("question_prompt")
    print("\nQuestion prompt content:\n", question_prompt_content)

    dataset_paths = config_loader.get_question_dataset_paths()
    print("Question dataset paths:", dataset_paths)

    visualization_dir = config_loader.get_visualization_dir()
    print("Visualization directory:", visualization_dir)