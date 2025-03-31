# testers/test_executor.py
import time
import os
from typing import List, Dict

from context_handlers.context_handler import ContextHandler
from utils.config_loader import ConfigLoader


class TestExecutor:
    """Executes tests for a given question, model, and context."""

    def __init__(
        self,
        config_path: str,
        question_prompt_template: str,
        context_handler: ContextHandler,
        output_dir: str,
    ):
        """
        Initializes the TestExecutor.

        Args:
            config_path (str): Path to the configuration file.
            evaluator (Evaluator): Evaluator instance for answer evaluation.
            question_prompt_template (str): Template for question prompts.
            context_handler (ContextHandler): Context handler instance.
            output_dir (str): Directory to save prompts for debugging.
        """
        self.config_loader = ConfigLoader(config_path)  # needed for output dir
        # self.evaluator = evaluator
        self.question_prompt_template = question_prompt_template
        self.context_handler = context_handler
        self.output_dir = output_dir
        self.timing_logs = []
        self.max_retries = 3
        self.retry_delay = 5

    def execute_test_question(
        self,
        question_data: Dict,
        context_type: str,
        noise_level: int,
        model_name: str,
        file_extension: str,
        llm_connector,
        results_list: List[Dict],
        language: str,
    ):
        """
        Executes a test for a single question.

        Args:
            question_data (Dict): Dictionary containing question details ('question', 'answer', 'page', 'dataset').
            context_type (str): 'page' or 'token'.
            noise_level (int): Noise level for context retrieval.
            model_name (str): Name of the LLM model being tested.
            file_extension (str): File extension of the document being tested.
            llm_connector: The LLM connector instance.
            results_list: List to append the result of the test to.
            language (str): Language of the test.

        Returns:
            Dict: The result of the test execution.
        """
        question_text = question_data["question"]
        expected_answer = question_data["answer"]
        target_page = question_data["page"]
        dataset = question_data.get("dataset", "unknown")

        context = self.context_handler.get_context(
            target_page, context_type, noise_level, file_extension
        )
        if not context:
            raise ValueError(f"Context is empty for page {target_page}")

        prompt = self.question_prompt_template.format(
            context=context, question=question_text
        )

        self._save_prompt_to_file(
            prompt, context_type, noise_level, file_extension, language
        )

        model_answer = "ERROR"
        retries = 0
        start_time = 0
        end_time = 0
        duration = 0

        while model_answer == "ERROR" and retries < self.max_retries:
            retries += 1
            start_time = time.time()
            try:
                model_answer = llm_connector.invoke(prompt)
            except Exception as e:
                print(
                    f"Error invoking model {model_name} (Retry {retries}/{self.max_retries}): {e}"
                )
                model_answer = "ERROR"
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
            finally:
                end_time = time.time()
                duration = end_time - start_time

        if model_answer == "ERROR":
            print(
                f"Max retries reached for question: {question_text}. Saving 'ERROR' as model answer."
            )


        result = {
            "question": question_text,
            "expected_answer": expected_answer,
            "model_answer": model_answer,
            "target_page": target_page,
            "context_type": context_type,
            "noise_level": noise_level,
            "model_name": model_name,
            "self_evaluation": None, # Set self_evaluation to None
            "dataset": dataset,
            "file_extension": file_extension,
            "duration": round(duration, 1),
        }
        results_list.append(result)  # Append to the provided list
        return result

    prompt_save_counter = 0  # Class-level counter

    def _save_prompt_to_file(
        self,
        prompt: str,
        context_type: str,
        noise_level: int,
        file_extension: str,
        language: str,
    ):
        """Saves the prompt to a file for debugging purposes."""
        TestExecutor.prompt_save_counter += 1
        if TestExecutor.prompt_save_counter % 25 == 0:
            os.makedirs(
                os.path.join(self.output_dir, "prompts"), exist_ok=True
            )  # Ensure prompts dir exists
            filename = (
                f"prompt_{language}_{file_extension}_{context_type}_{noise_level}.txt"
            )
            path_to_write_prompt = os.path.join(
                self.output_dir,
                "prompts",
                filename,
            )
            with open(path_to_write_prompt, "w", encoding="utf-8") as f:
                f.write(prompt)
