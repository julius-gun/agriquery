# RAG/utils/config_loader.py
import json
import os
import re
from typing import List, Dict, Any  # Import List, Dict, Any


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
            # In a real scenario, you might want to define actual defaults here
            # or raise an error if the config is essential.
            # For now, returning an empty dict might suffice for some getters,
            # but others will fail. Let's keep the original default loader.
            return self._load_default_config()  # Keep default loading logic
        except json.JSONDecodeError:
            raise ValueError(
                f"Error decoding JSON from '{self.config_path}'. Please check if the file is valid JSON."
            )

    def _load_default_config(self):
        default_config = {
            {
                "llm_models": {
                    "ollama": {
                        "deepseek-r1_8B-128k": {
                            "name": "deepseek-r1:8b",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 128000,
                        },
                        "deepseek-r1_1.5B-128k": {
                            "name": "deepseek-r1:1.5b",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 128000,
                        },
                        "deepseek-r1_14B-128k": {
                            "name": "deepseek-r1:14b",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 128000,
                        },
                        "qwen2.5_7B-128k": {
                            "name": "qwen2.5:latest",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 128000,
                        },
                        "phi3_14B_q4_medium-128k": {
                            "name": "phi3:medium-128k",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 128000,
                        },
                        "phi3_8B_q4_mini-128k": {
                            "name": "phi3:mini-128k",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 128000,
                        },
                        "llama3.1_8B-128k": {
                            "name": "llama3.1:latest",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 128000,
                        },
                        "llama3.2_3B-128k": {
                            "name": "llama3.2:latest",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 128000,
                        },
                        "llama3.2_1B-128k": {
                            "name": "llama3.2:1b",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 128000,
                        },
                        "phi3_14B_medium-4k": {
                            "name": "phi3:medium",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 4000,
                        },
                        "gemma2_9B-8k": {
                            "name": "gemma2:latest",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 8192,
                        },
                        "gemma3_12B-128k": {
                            "name": "gemma3:12b",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 128000,
                        },
                        "phi4_14B-16k": {
                            "name": "phi4:latest",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 16384,
                        },
                        "qwq_32B-128k": {
                            "name": "qwq",
                            "temperature": 0.0,
                            "num_predict": 1024,
                            "context_window": 131072,
                        },
                    }
                },
                "question_models_to_test": [
                    "deepseek-r1_8B-128k",
                    "deepseek-r1_1.5B-128k",
                    "deepseek-r1_14B-128k",
                    "qwen2.5_7B-128k",
                    "phi3_14B_q4_medium-128k",
                    "phi3_8B_q4_mini-128k",
                    "llama3.1_8B-128k",
                    "llama3.2_3B-128k",
                    "llama3.2_1B-128k",
                    "phi3_14B_medium-4k",
                    "phi4_14B-16k",
                    "qwq_32B-128k",
                ],
                "evaluator_model_name": "gemma3_12B-128k",
                "prompt_paths": {
                    "question_prompt": "prompt_templates/question_prompt.txt",
                    "evaluation_prompt": "prompt_templates/evaluation_prompt.txt",
                },
                "language_configs": [
                    {
                        "language": "english",
                        "manual_path": "manuals/english_manual.txt",
                        "collection_base_name": "english_manual",
                    },
                    {
                        "language": "french",
                        "manual_path": "manuals/french_manual.txt",
                        "collection_base_name": "french_manual",
                    },
                    {
                        "language": "german",
                        "manual_path": "manuals/german_manual.txt",
                        "collection_base_name": "german_manual",
                    },
                ],
                "question_dataset_paths": {
                    "general_questions": "question_datasets/question_answers_pairs.json",
                    "table_questions": "question_datasets/question_answers_tables.json",
                    "unanswerable_questions": "question_datasets/question_answers_unanswerable.json",
                },
                "output_dir": "results",
                "rag_parameters": {
                    "retrieval_algorithms_to_test": ["embedding"],
                    "num_retrieved_docs": 3,
                    "chunk_size": 2000,
                    "overlap_size": 50,
                },
            }
        }
        return default_config

    def get_llm_models_config(self, llm_type="ollama") -> Dict[str, Any]:
        llm_models = self.config.get("llm_models", {})
        models_for_type = llm_models.get(llm_type.lower(), {})
        if not models_for_type:
            # Return empty dict or raise error depending on desired strictness
            print(
                f"Warning: LLM type '{llm_type}' not found or has no models defined in config."
            )
            # raise ValueError(f"LLM type '{llm_type}' not supported or configured in config.")
        return models_for_type

    def get_question_model_name(self) -> str | None:
        """Gets the single question model name (potentially a default)."""
        return self.config.get("question_model_name")

    def get_question_models_to_test(self) -> List[str]:
        """Gets the list of question model names to iterate through for testing."""
        models = self.config.get("question_models_to_test", [])
        if not isinstance(models, list):
            print(
                f"Warning: 'question_models_to_test' in config is not a list. Found: {type(models)}. Returning empty list."
            )
            return []
        return models

    def get_evaluator_model_name(self) -> str | None:
        return self.config.get("evaluator_model_name")

    def get_prompt_path(self, prompt_name: str) -> str:
        prompt_paths = self.config.get("prompt_paths", {})
        path = prompt_paths.get(prompt_name)
        if not path:
            raise ValueError(f"Prompt template '{prompt_name}' not defined in config.")
        return path

    def get_question_dataset_paths(self) -> Dict[str, str]:
        return self.config.get("question_dataset_paths", {})

    def get_output_dir(self) -> str:
        return self.config.get("output_dir", "results")

    def load_prompt_template(self, prompt_name: str) -> str:
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

    def get_rag_parameters(self) -> Dict[str, Any]:
        # Return the whole dictionary, default to empty if not found
        return self.config.get("rag_parameters", {})

    def get_retrieval_algorithms_to_test(self) -> List[str]:
        """Gets the list of retrieval algorithm names to iterate through for testing."""
        rag_params = self.get_rag_parameters()  # Use the existing getter
        algorithms = rag_params.get("retrieval_algorithms_to_test", [])
        if not isinstance(algorithms, list):
            print(
                f"Warning: 'retrieval_algorithms_to_test' in rag_parameters is not a list. Found: {type(algorithms)}. Returning empty list."
            )
            return []
        return algorithms


if __name__ == "__main__":
    config_loader = ConfigLoader()  # Load default or config.json

    print("--- Testing ConfigLoader ---")

    # Test existing methods
    ollama_models = config_loader.get_llm_models_config("ollama")
    print("\nOllama models:", json.dumps(ollama_models, indent=2))

    evaluator_model = config_loader.get_evaluator_model_name()
    print("\nEvaluator model name:", evaluator_model)

    # Test new methods
    question_models_list = config_loader.get_question_models_to_test()
    print("\nQuestion models to test:", question_models_list)

    retrieval_algorithms_list = config_loader.get_retrieval_algorithms_to_test()
    print("\nRetrieval algorithms to test:", retrieval_algorithms_list)

    # Test other methods
    question_prompt_path = config_loader.get_prompt_path("question_prompt")
    print("\nQuestion prompt path:", question_prompt_path)

    try:
        question_prompt_content = config_loader.load_prompt_template("question_prompt")
        print(
            "\nQuestion prompt content loaded successfully."
        )  # Don't print content itself
    except Exception as e:
        print(f"\nError loading question prompt content: {e}")

    dataset_paths = config_loader.get_question_dataset_paths()
    print("\nQuestion dataset paths:", dataset_paths)

    rag_parameters = config_loader.get_rag_parameters()
    print("\nRAG parameters:", rag_parameters)

    output_dir = config_loader.get_output_dir()
    print("\nOutput directory:", output_dir)

    print("\n--- ConfigLoader Test Finished ---")
