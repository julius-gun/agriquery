# llm_tester.py
import json
import os
from typing import List
from tqdm import tqdm
import time
from context_handlers.context_handler import ContextHandler
from document_loaders.document_loader import DocumentLoader
from evaluation.evaluator import Evaluator
from llm_connectors.ollama_connector import OllamaConnector
from utils.config_loader import ConfigLoader
from utils.metrics import calculate_metrics
from utils.question_tracker import (
    QuestionTracker,
)  # Keep for compatibility, even if not directly used
from utils.question_loader import QuestionLoader
from llm_connectors.llm_connector_manager import LLMConnectorManager
from testers.test_executor import TestExecutor
from utils.result_manager import ResultManager


class LLMTester:
    """Orchestrates LLM testing, now using modular components."""

    def __init__(self, config_path: str = "config.json"):
        self.config_loader = ConfigLoader(config_path)
        self.output_dir = self.config_loader.get_output_dir()
        self.document_loader = DocumentLoader(self.output_dir, config_path)
        self.question_tracker = QuestionTracker(
            self.config_loader.get_question_tracker_path()
        )
        self.question_prompt_template = self.config_loader.load_prompt_template(
            "question_prompt"
        )
        self.evaluation_prompt_template = self.config_loader.load_prompt_template(
            "evaluation_prompt"
        )
        self.file_extensions_to_test = self.config_loader.config.get(
            "file_extensions_to_test", ["txt"]
        )

        # Initialize modular components - Evaluator initialization REMOVED from here
        self.question_loader = QuestionLoader(
            self.config_loader.get_question_dataset_paths()
        )
        self.llm_connector_manager = LLMConnectorManager(
            self.config_loader.config["llm_models"]
        )
        self.result_manager = ResultManager(self.output_dir)

    def is_result_file_complete(
        self, language, model_name, file_extension, context_type, noise_level
    ):
        """
        Checks if the result file for a given configuration is 100% answered.
        """
        results = self.result_manager.load_previous_results(
            language, model_name, file_extension, context_type, noise_level
        )
        questions = (
            self.question_loader.load_questions()
        )  # Load questions to count them
        total_questions = len(questions)
        num_results = len(results)

        return (
            num_results >= total_questions
        )  # Consider >= in case of previous runs having more results for some reason

    def run_tests(
        self,
        llm_model_name: str,
        llm_type: str,
        context_type: str,
        noise_levels: List[int],
        documents_to_test,
        models_to_test: List[str] = None, # keep for compatibility, even if not used here directly
    ):
        """Runs tests for the specified LLM and configurations, using modular components."""

        questions = self.question_loader.load_questions()
        # total_questions = len(questions)
        current_llm_type = llm_type
        model_name = llm_model_name

        llm_connector = self.llm_connector_manager.get_connector(  # Get connector via manager - load model only once per run_tests call
            current_llm_type, model_name
        )
        print(
            f"DEBUG: Loaded connector for model: {model_name} ({current_llm_type})"
        )


        for document_config in documents_to_test:  # Outer loop: Documents
            document_url = document_config["url"]
            local_filename = document_config["local_filename"]
            language = document_config["language"]

            with tqdm(
                total=len(self.file_extensions_to_test),
                desc=f"Extensions for {model_name} - {local_filename}",
            ) as extension_pbar:
                for file_extension in (
                    self.file_extensions_to_test
                ):  # Middle loop 1: File Extensions for testing
                    # Re-initialize document loader and context handler for each document/model/extension combination
                    self.document_loader = DocumentLoader(
                        self.config_loader.get_output_dir(),
                        self.config_loader.config_path,
                        local_filename,
                    )
                    self.pages = self.document_loader.load_document(document_url)
                    if not self.pages:
                        print(
                            f"No pages loaded for {document_url}, skipping {model_name}, {file_extension}"
                        )
                        extension_pbar.update(1)
                        continue  # Skip to the next extension if no pages are loaded
                    self.context_handler = ContextHandler(
                        self.pages, self.config_loader.config_path
                    )

                    with tqdm(
                        total=len(noise_levels), desc=f"Noise Levels ({file_extension})"
                    ) as noise_level_pbar:
                        for noise_level in noise_levels:  # Middle loop 2: Noise Levels
                            if self.is_result_file_complete(language, model_name, file_extension, context_type, noise_level):
                                # print(f"Results already 100% for {language.capitalize()} manual: {model_name} - Noise {noise_level} - {file_extension}. Skipping tests.")
                                noise_level_pbar.update(1) # Still update progress bar
                                continue # Skip to the next noise level

                            print(f"Starting tests for language: {language}")
                            print(
                                f"Running tests with {language.capitalize()} manual: {model_name} ({current_llm_type}), context type: {context_type}, noise level: {noise_level}, file extension: {file_extension}"
                            )

                            # Load previous results *before* the question loop  using ResultManager
                            self.results = self.result_manager.load_previous_results(
                                language,
                                model_name,
                                file_extension,
                                context_type,
                                noise_level,
                            )

                            unanswered_questions = []
                            for question_data in questions:
                                question_text = question_data["question"]
                                already_answered = any(
                                    r["question"] == question_text
                                    and r["context_type"] == context_type
                                    and r["noise_level"] == noise_level
                                    and r["model_name"] == model_name
                                    and r["file_extension"] == file_extension
                                    for r in self.results
                                )
                                if not already_answered:
                                    unanswered_questions.append(question_data)

                            num_unanswered_questions = len(unanswered_questions)
                            # print(
                            #     f"Number of unanswered questions: {num_unanswered_questions}"
                            # )

                            if num_unanswered_questions > 0:

                                test_executor = TestExecutor(  # Initialize TestExecutor here
                                    self.config_loader.config_path,
                                    # self.evaluator, # Removed: Evaluator not needed in TestExecutor during testing
                                    self.question_prompt_template,
                                    self.context_handler,
                                    self.output_dir,
                                )

                                # Initialize tqdm *inside* the noise_level loop, only if there are unanswered questions
                                question_pbar = tqdm(
                                    total=num_unanswered_questions,
                                    desc=f"Testing {language.capitalize()} manual: {model_name} - Noise {noise_level} - {file_extension}",
                                    leave=True,
                                )
                                # Refresh progress bar immediately after initialization to ensure it's displayed correctly
                                question_pbar.refresh()

                                for (
                                    question_data
                                ) in unanswered_questions:  # Inner loop: Questions
                                    question_text = question_data["question"]

                                    try:
                                        test_result = test_executor.execute_test_question(  # Use TestExecutor to run test  # noqa: F841
                                            question_data,
                                            context_type,
                                            noise_level,
                                            model_name,
                                            file_extension,
                                            llm_connector,
                                            self.results,  # Pass self.results to append to
                                            language,  # Pass language to test executor
                                        )

                                        self.result_manager.save_results(  # Save results using ResultManager
                                            self.results,
                                            language,
                                            model_name,
                                            file_extension,
                                            context_type,
                                            noise_level,
                                        )

                                    except Exception as question_error:
                                        print(
                                            f"Error processing question: {question_text}. Error: {question_error}"
                                        )
                                    finally:
                                        question_pbar.update(1)  # ALWAYS update

                                question_pbar.close()  # Close the progress bar


                            else:
                                print(
                                    f"All questions already answered for {language.capitalize()} manual: {model_name} - Noise {noise_level} - {file_extension}. Skipping tests."
                                )
                            noise_level_pbar.update(
                                1
                            )  # Update noise level progress bar
                    extension_pbar.update(1)  # Update extension progress bar
            if current_llm_type == "ollama":
                try:
                    print(f"DEBUG: Stopping model: {model_name}")
                    self.stop_model(
                        model_name
                    )  # Use model_name as model_id for ollama
                    time.sleep(
                        5
                    )  # Add a 5-second delay after stopping # ADD THIS LINE
                except Exception as e:
                    print(f"Error stopping model {model_name}: {e}")
            # print(f"Completed tests for model: {model_name}")

    def run_evaluations(
        self,
        llm_model_name: str,
        file_extensions_to_test: List[str],
        noise_levels: List[int],
        context_type: str,
        documents_to_test, # ADD documents_to_test here
    ):
        """
        Runs evaluations for the specified models and configurations in a separate loop.
        """
        # Initialize Evaluator - Moved here, to be initialized only when needed for evaluation
        evaluator_model_name = self.config_loader.get_evaluator_model_name()
        evaluator_config = self.config_loader.get_llm_models_config("ollama").get(
            evaluator_model_name, {"name": evaluator_model_name}
        )
        if "name" not in evaluator_config:
            evaluator_config["name"] = evaluator_model_name
        evaluator_connector = OllamaConnector(evaluator_model_name, evaluator_config)
        self.evaluator = Evaluator(evaluator_connector, self.evaluation_prompt_template)
        model_name = llm_model_name

        for document_config in documents_to_test: # Iterate through documents
            language = document_config["language"] # Get language from document config
            with tqdm(
                total=len(file_extensions_to_test),
                desc=f"Evaluating Extensions for {model_name} - {language}", # Add language to description
            ) as extension_pbar:
                for file_extension in (
                    file_extensions_to_test
                ):  # Middle loop 1: File Extensions for evaluation
                    with tqdm(
                        total=len(noise_levels),
                        desc=f"Evaluating Noise Levels ({file_extension})",
                    ) as noise_level_pbar:
                        for (
                            noise_level
                        ) in noise_levels:  # Middle loop 2: Noise Levels for evaluation
                            print(
                                f"Starting deferred evaluation for {language.capitalize()} manual: {model_name} - {file_extension} - Noise {noise_level}"
                            )
                            self.evaluate_results(
                                language, # Use the language from the current document
                                model_name,
                                file_extension,
                                context_type,
                                [noise_level],
                            )  # Call evaluate_results here, for each noise level

                            noise_level_pbar.update(
                                1
                            )  # Update noise level progress bar
                    extension_pbar.update(1)  # Update extension progress bar
        print(f"Completed evaluations for model: {model_name}")

    def evaluate_results(
        self, language, model_name, file_extension, context_type, noise_levels
    ):
        """
        Evaluates model answers for results with empty self_evaluation.
        Now accepts noise_levels list to process results for all noise levels in one go.
        """
        all_results = []
        for noise_level in noise_levels:  # Iterate over noise levels to evaluate all results for current model, extension, context_type
            results_dir = os.path.join(self.output_dir)
            filename = self.result_manager._generate_filename(language, model_name, file_extension, context_type, noise_level) # FIXED FILENAME GENERATION
            filepath = os.path.join(results_dir, filename) # FIXED FILENAME GENERATION


            if not os.path.exists(filepath):
                print(
                    f"Results file not found: {filepath}. Skipping evaluation for noise level {noise_level}."
                )
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                results = json.load(f)
                all_results.extend(results)  # Collect results from all noise levels

        if not all_results:
            print(
                f"No results found for evaluation for {model_name}, {file_extension}, {context_type}, noise levels {noise_levels}."
            )
            return

        num_to_evaluate = sum(
            1 for res in all_results if res.get("self_evaluation") is None
        )
        if num_to_evaluate == 0:
            print(
                f"No results to evaluate for {model_name}, {file_extension}, {context_type}, noise levels {noise_levels}."
            )
            return

        print(
            f"Evaluating {num_to_evaluate} results for {model_name}, {file_extension}, {context_type}, noise levels {noise_levels}..."
        )
        evaluation_pbar = tqdm(
            total=num_to_evaluate,
            desc=f"Evaluating {model_name} - {file_extension} - {context_type}",
            leave=True,
        )

        for result in all_results:  # Evaluate all collected results
            if result.get("self_evaluation") is None:
                # print(
                #     f"DEBUG: Found result to evaluate for question: {result['question'][:50]}..., model: {model_name}, noise level: {noise_levels}"
                # ) # DEBUG PRINT
                question_text = result["question"]
                model_answer = result["model_answer"]
                expected_answer = result["expected_answer"]

                evaluation_result = self.evaluator.evaluate_answer(
                    question_text, model_answer, expected_answer
                )
                result["self_evaluation"] = evaluation_result
                evaluation_pbar.update(1)

        evaluation_pbar.close()

        # Save results back to individual files per noise level
        for noise_level in noise_levels:
            results_to_save = [
                res for res in all_results if res["noise_level"] == noise_level
            ]  # Filter results for current noise level
            results_dir = os.path.join(self.output_dir)
            # filename = f"{language}_{model_name}_{file_extension}_{context_type}_{noise_level}_results.json" # OLD FILENAME GENERATION
            # filename = f"{language}_{model_name}_{file_extension}_{context_type}_{noise_level}_results.json" # FIXED FILENAME GENERATION HERE TOO
            filename = self.result_manager._generate_filename(language, model_name, file_extension, context_type, noise_level) # FIXED FILENAME GENERATION HERE TOO
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    results_to_save, f, indent=4, ensure_ascii=False
                )  # Save updated results for each noise level

        print(
            f"Evaluation completed and saved for {model_name}, {file_extension}, {context_type}, noise levels {noise_levels}."
        )
        self.results = all_results  # Update self.results to use for metrics calculation - using all_results now
        self.calculate_and_display_metrics()  # Calculate and display metrics after evaluation

    def calculate_and_display_metrics(self):
        """Calculates and displays overall metrics."""
        overall_metrics = calculate_metrics(self.results)
        # create_duration_boxplot(
        #     self.results,
        #     self.config_loader.get_output_dir(),
        #     filename_prefix="duration_boxplot",
        # )
        print("\nOverall Metrics:")
        for metric, value in overall_metrics.items():
            print(f"{metric.capitalize()}: {value:.2f}")

        dataset_metrics = {}
        for result in self.results:
            dataset = result.get("dataset", "unknown")
            if dataset not in dataset_metrics:
                dataset_metrics[dataset] = []
            dataset_metrics[dataset].append(result)

        print("\nMetrics per Dataset:")
        for dataset, results in dataset_metrics.items():
            metrics = calculate_metrics(results)
            print(f"\nDataset: {dataset}")
            for metric, value in metrics.items():
                print(f"  {metric.capitalize()}: {value:.2f}")
