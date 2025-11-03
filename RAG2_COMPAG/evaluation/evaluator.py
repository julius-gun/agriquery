from llm_connectors.base_llm_connector import BaseLLMConnector
from utils.config_loader import ConfigLoader

class Evaluator:
    """Evaluates model answers using another LLM."""

    def __init__(self, evaluator_model_connector: BaseLLMConnector, evaluation_prompt_template: str):
        """
        Initializes the Evaluator.

        Args:
            evaluator_model_connector (BaseLLMConnector): Connector for the LLM used as evaluator.
            evaluation_prompt_template (str): Prompt template for evaluation.
        """
        self.evaluator_model_connector = evaluator_model_connector
        self.evaluation_prompt_template = evaluation_prompt_template

    def evaluate_answer(self, question: str, model_answer: str, expected_answer: str) -> str:
        """
        Evaluates the model's answer against the expected answer using the evaluator LLM.

        Args:
            question (str): The question asked.
            model_answer (str): The answer provided by the model being evaluated.
            expected_answer (str): The expected correct answer.

        Returns:
            str: The evaluation result (e.g., "yes" or "no").
        """
        eval_prompt = self.evaluation_prompt_template.format(
            question=question,
            model_answer=model_answer,
            expected_answer=expected_answer
        )
        evaluation_response = self.evaluator_model_connector.invoke(eval_prompt).strip().lower()
        return evaluation_response



if __name__ == '__main__':
    # Example usage (requires a config and an LLM connector - using Ollama example):
    config_loader = ConfigLoader()
    evaluator_model_name = config_loader.get_evaluator_model_name() # e.g., "gemma2:latest"
    if not evaluator_model_name:
        raise ValueError("Evaluator model name not found in config. Cannot run example.")

    # Safely get the ollama config, defaulting to an empty dict if the 'ollama' section is missing
    ollama_models_config = config_loader.get_llm_models_config("ollama") or {}
    # Get config for the specific evaluator model, or provide a default
    ollama_config = ollama_models_config.get(evaluator_model_name, {"name": evaluator_model_name})

    if "name" not in ollama_config:
        ollama_config["name"] = evaluator_model_name # Ensure name is set

    from llm_connectors.ollama_connector import OllamaConnector
    evaluator_connector = OllamaConnector(evaluator_model_name, ollama_config) # Using Ollama as evaluator in this example

    evaluation_prompt_template_content = config_loader.load_prompt_template("evaluation_prompt")
    evaluator = Evaluator(evaluator_connector, evaluation_prompt_template_content)

    question_example = "What is the capital of France?"
    model_answer_example = "Paris"
    expected_answer_example = "Paris"

    evaluation_result = evaluator.evaluate_answer(question_example, model_answer_example, expected_answer_example)
    print(f"Question: {question_example}")
    print(f"Model Answer: {model_answer_example}")
    print(f"Expected Answer: {expected_answer_example}")
    print(f"Evaluation Result: {evaluation_result}") # Expected: yes (or similar)

    # Example with a wrong answer
    wrong_answer_result = evaluator.evaluate_answer(question_example, "London", expected_answer_example)
    print(f"\nEvaluation Result for wrong answer: {wrong_answer_result}") # Expected: no (or similar)
