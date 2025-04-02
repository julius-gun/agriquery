# RAG/utils/config_loader.py
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
            },
            "question_model_name": "qwen2.5_7B-128k",
            "evaluator_model_name": "gemma2_9B-8k",
            "prompt_paths": {
                "question_prompt": "RAG/prompt_templates/question_prompt.txt",
                "evaluation_prompt": "RAG/prompt_templates/evaluation_prompt.txt",
            },
            "question_dataset_paths": {
                "general_questions": "question_datasets/question_answers_pairs.json",
                "table_questions": "question_datasets/question_answers_tables.json",
                "unanswerable_questions": "question_datasets/question_answers_unanswerable.json"
             },
            "output_dir": "results",
            "rag_parameters": {  # Added rag_parameters section
                "retrieval_algorithm": "embedding",
                "num_retrieved_docs": 3,
            },
        }
        return default_config

    def get_llm_models_config(self, llm_type="ollama"):
        llm_type = llm_type.lower()
        if llm_type not in self.config["llm_models"]:
            raise ValueError(f"LLM type '{llm_type}' not supported in config.")
        return self.config["llm_models"][llm_type]

    def get_question_model_name(self):
        return self.config[
            "question_model_name"
        ]  # Changed to get from top-level config

    def get_evaluator_model_name(self):
        return self.config["evaluator_model_name"]

    def get_prompt_path(self, prompt_name):
        if prompt_name not in self.config["prompt_paths"]:
            raise ValueError(f"Prompt template '{prompt_name}' not defined in config.")
        return self.config["prompt_paths"][prompt_name]

    def get_question_dataset_paths(self):
        return self.config["question_dataset_paths"]

    def get_output_dir(self):
        return self.config.get("output_dir", "results") # default output dir is "results" if not in config

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

    def get_rag_parameters(self):
        return self.config["rag_parameters"]


if __name__ == "__main__":
    config_loader = ConfigLoader()

    ollama_models = config_loader.get_llm_models_config("ollama")
    print("Ollama models:", ollama_models)

    evaluator_model = config_loader.get_evaluator_model_name()
    print("Evaluator model name:", evaluator_model)

    question_model = config_loader.get_question_model_name()
    print("Question model name:", question_model)

    question_prompt_path = config_loader.get_prompt_path("question_prompt")
    print("Question prompt path:", question_prompt_path)

    question_prompt_content = config_loader.load_prompt_template("question_prompt")
    print("\nQuestion prompt content:\n", question_prompt_content)

    dataset_paths = config_loader.get_question_dataset_paths()
    print("Question dataset paths:", dataset_paths)

    rag_parameters = config_loader.get_rag_parameters()
    print("RAG parameters:", rag_parameters)
    output_dir = config_loader.get_output_dir()
    print("Output directory:", output_dir)