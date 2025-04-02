# RAG/testers/rag_tester.py
import json
import os
import time,  datetime

from analysis.analysis_tools import analyze_evaluation_results, analyze_dataset_across_types, load_dataset, calculate_and_analyze_metrics
from evaluation.evaluator import Evaluator
from llm_connectors.llm_connector_manager import LLMConnectorManager
from parameter_tuning.parameters import RagParameters
from rag_pipeline import initialize_retriever, collection  # Assuming collection and initialize_retriever are needed
from utils.config_loader import ConfigLoader
from evaluation.metrics import calculate_metrics # Import calculate_metrics


def run_rag_test(config_path="config.json"): # Changed default config path to config.json
    """
    Runs the RAG test based on the provided configuration.
    """
    config_loader = ConfigLoader(config_path)
    rag_config = config_loader.config # Load the entire config

    # --- Load Models ---
    llm_connector_manager = LLMConnectorManager(config_loader.get_llm_models_config("ollama")) # Pass only ollama config
    question_model_name = config_loader.get_question_model_name() # Use getter
    question_llm_config = config_loader.get_llm_models_config("ollama").get(question_model_name, {"name": question_model_name}) # get model config
    question_llm_connector = llm_connector_manager.get_connector("ollama", question_model_name) # Assuming ollama for now

    evaluator_model_name = config_loader.get_evaluator_model_name()
    evaluator_llm_config = config_loader.get_llm_models_config("ollama").get(evaluator_model_name, {"name": evaluator_model_name}) # get model config for evaluator model
    evaluator_llm_connector = llm_connector_manager.get_connector("ollama", evaluator_model_name) # Assuming ollama for now


    # --- Load Prompts ---
    question_prompt_template = config_loader.load_prompt_template("question_prompt")
    evaluation_prompt_template = config_loader.load_prompt_template("evaluation_prompt")

    # --- Initialize Evaluator ---
    evaluator = Evaluator(evaluator_llm_connector, evaluation_prompt_template)

    # --- Load Datasets ---
    dataset_paths = config_loader.get_question_dataset_paths() # Use getter for dataset paths
    all_results = {}

    for dataset_name, dataset_path in dataset_paths.items():
        dataset = load_dataset(dataset_path)
        if not dataset:
            print(f"Skipping dataset: {dataset_name} due to loading error.")
            continue

        dataset_results = []
        start_time = time.time()

        for question_data in dataset:
            question = question_data["question"]
            expected_answer = question_data["answer"]
            page = question_data["page"] # Assuming page number is available

            # --- RAG Retrieval ---
            rag_params = config_loader.get_rag_parameters() # Get RAG parameters
            retriever = initialize_retriever(rag_params.get("retrieval_algorithm", "embedding")) # Initialize retriever from config
            question_embedding = retriever.vectorize_text(question)

            results = collection.get(include=['embeddings', 'documents']) # Fetch all embeddings and documents - to be optimized later if needed
            document_chunk_embeddings_from_db = results['embeddings']
            document_chunks_text_from_db = results['documents']

            retrieved_chunks_text, _ = retriever.retrieve_relevant_chunks(
                question_embedding,
                document_chunk_embeddings_from_db,
                document_chunks_text_from_db,
                top_k=rag_params.get("num_retrieved_docs", 3) # Retrieve top_k documents as per RAG params
            )
            context = "\n".join(retrieved_chunks_text) # Concatenate retrieved chunks into context string


            # --- LLM Question Answering ---
            prompt = question_prompt_template.format(context=context, question=question)
            model_answer = question_llm_connector.invoke(prompt)


            # --- Evaluation ---
            evaluation_result = evaluator.evaluate_answer(question, model_answer, expected_answer)
            print(f"Evaluation Result: {evaluation_result}") # Debug print

            result_entry = {
                "question": question,
                "expected_answer": expected_answer,
                "model_answer": model_answer,
                "self_evaluation": evaluation_result, # Changed 'evaluation' to 'self_evaluation' and store evaluator result
                "page": page,
                "dataset": dataset_name,
            }
            print(f"Dataset Name: {dataset_name}") # Debug print
            print(f"Result Entry: {result_entry}") # Debug print
            dataset_results.append(result_entry)

        end_time = time.time()
        dataset_duration = end_time - start_time

        print(f"Dataset Results before metrics calculation: {dataset_results}") # Debug print
        evaluation_metrics = calculate_and_analyze_metrics(dataset_results, dataset_name) # Calculate and analyze metrics

        dataset_evaluation_results = { # Use the metrics dictionary directly
            "metrics": evaluation_metrics,
            "total_questions": len(dataset_results),
            "duration": dataset_duration
        }


        all_results[dataset_name] = {
            "results": dataset_results,
            "evaluation_summary": dataset_evaluation_results
        }
        analyze_evaluation_results(dataset_evaluation_results["metrics"], dataset_name) # Pass metrics to analysis function


    # --- Overall Analysis ---
    analyze_dataset_across_types(dataset_paths) # Analyze across datasets
    print("\n--- Overall RAG Test Analysis ---")
    for dataset_name, result_data in all_results.items():
        summary = result_data["evaluation_summary"]
        metrics = summary["metrics"] # Get metrics from summary
        print(f"Dataset: {dataset_name}, Accuracy: {metrics['accuracy']:.2f}%, Precision: {metrics['precision']:.2f}%, Recall: {metrics['recall']:.2f}%, F1 Score: {metrics['f1_score']:.2f}%, Duration: {summary['duration']:.2f}s")


    # --- Save Results to JSON ---
    output_dir = config_loader.get_output_dir() # Use getter for output dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # Use more robust datetime for timestamp
    results_filename = os.path.join(output_dir, f"rag_test_results_{timestamp}.json")
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to: {results_filename}")


if __name__ == "__main__":
    run_rag_test()