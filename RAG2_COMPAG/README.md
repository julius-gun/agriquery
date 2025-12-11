# Agri-Query RAG Testing Framework

**Quick Links:**
*   📂 **RAG Results (JSON):** [results/](results/)
*   📊 **Visualizations (Plots):** [visualization/plots/](visualization/plots/)
*   📋 **LaTeX Tables:** [visualization/latex_tables/](visualization/latex_tables/)

---

## 📈 Visualizations & Results

### 1. Global Overview (Model vs Language)
Performance heatmaps combining all formats (JSON, Markdown, XML).

![F1 Score Overview](visualization/plots/heatmap_language_vs_model_f1_score_combined.png)
![Accuracy Overview](visualization/plots/heatmap_language_vs_model_accuracy_combined.png)

### 2. Format Comparison
Impact of data format on RAG performance.

**F1 Score:**
![Format F1 (Avg)](visualization/plots/heatmap_format_vs_model_f1_score_avg_all_langs.png)
![Format F1 (English)](visualization/plots/heatmap_format_vs_model_f1_score_english.png)

**Accuracy:**
![Format Accuracy (Avg)](visualization/plots/heatmap_format_vs_model_accuracy_avg_all_langs.png)
![Format Accuracy (English)](visualization/plots/heatmap_format_vs_model_accuracy_english.png)

#### Detailed Bar Charts (Format Performance)
**Average (All Languages):**
![Format F1 (Avg)](visualization/plots/barcharts/hybrid_format_perf_f1_score_avg_all_langs.png)
![Format Accuracy (Avg)](visualization/plots/barcharts/hybrid_format_perf_accuracy_avg_all_langs.png)

**English Only:**
![Format F1 (English)](visualization/plots/barcharts/hybrid_format_perf_f1_score_english.png)
![Format Accuracy (English)](visualization/plots/barcharts/hybrid_format_perf_accuracy_english.png)

### 3. Cross-Lingual Capabilities
Comparison of English performance vs. Average Non-English performance.

![Cross Lingual F1](visualization/plots/scatterplots/scatter_cross_lingual_f1_score_combined.png)
![Cross Lingual Accuracy](visualization/plots/scatterplots/scatter_cross_lingual_accuracy_combined.png)

### 4. Generated Tables
The following LaTeX tables contain detailed metrics for each format:
*   📄 **JSON:** [tables_hybrid_json.tex](visualization/latex_tables/tables_hybrid_json.tex)
*   📄 **Markdown:** [tables_hybrid_md.tex](visualization/latex_tables/tables_hybrid_md.tex)
*   📄 **XML:** [tables_hybrid_xml.tex](visualization/latex_tables/tables_hybrid_xml.tex)

---

## Project Overview

To convert a PDF file into txt for this framework, please use the docling_page_wise_pdf_converter inside the folder `zeroshot\docling_page_wise_pdf_converter`.

This project provides a framework for building, testing, and evaluating Retrieval-Augmented Generation (RAG) pipelines. It allows testing different language manuals against a common set of English questions, using configurable LLMs, retrieval parameters, and evaluation metrics.

## Features

*   **Multi-Language Support:** Process and query documents in different languages (configured in `config.json`).
*   **Configurable Components:** Easily configure LLMs (via Ollama), vector database parameters (chunk size, overlap), retrieval settings (number of documents), and file paths through `config.json`.
*   **Vector Database Integration:** Uses ChromaDB for persistent vector storage. Includes scripts to create databases based on specific chunking strategies.
*   **Automated Testing:** Runs English question datasets against specified language document collections.
*   **LLM-Based Evaluation:** Employs a separate LLM to evaluate the correctness of the generated answers against expected answers.
*   **Performance Metrics:** Calculates Accuracy, Precision, Recall, Specificity, and F1-Score based on evaluation results.
*   **Detailed Results:** Saves comprehensive test results, including parameters, timings, metrics, and individual question/answer details, to JSON files.
*   **Docker Support:** Includes a Dockerfile for containerized setup and execution, managing Ollama service and dependencies.

## Prerequisites

*   Python 3.x (Tested with 3.12)
*   Docker (Recommended for managing Ollama and dependencies)
*   Git
*   Ollama installed and running (either locally or via the provided Docker setup).
*   **Required Ollama Models:** Ensure the LLMs specified in `config.json` (`question_models_to_test`, `evaluator_model_name`) are pulled in your Ollama instance (e.g., `ollama pull llama3.2:3b`, `ollama pull gemma3:12b`). The Dockerfile attempts to pull several models during the build process.
*   **Embedding Model:** The system relies on the embedding model specified in `config.json` (e.g., `Qwen/Qwen3-Embedding-8B`). `pip install -r requirements.txt` handles installing the necessary libraries.

## Docker Installation

*Note: Instructions for setting up and running with Docker can be found in the [DOCKER_README.md](Docker-README.md). The provided `Dockerfile` aims to set up the environment and pull necessary models.*

## Manual Installation

For manual installation, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd RAG2_COMPAG/
    ```

2.  **Create a virtual environment:**

    It is recommended to use a virtual environment to isolate project dependencies.

    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**

    *   **On Windows:**

        ```bash
        .venv\Scripts\activate
        ```

    *   **On macOS and Linux:**

        ```bash
        source .venv/bin/activate
        ```

    **Setting up a Virtual Environment in VS Code:**

    Visual Studio Code (VS Code) can automatically detect and use virtual environments. Here’s how to ensure your virtual environment is correctly set up in VS Code:

    1.  **Open the Project in VS Code:** Open the `RAG2_COMPAG` folder in VS Code.

    2.  **VS Code should detect the virtual environment:** When you open the project, VS Code should automatically detect the `.venv` virtual environment in your project directory. You might see a notification in the bottom right corner of VS Code suggesting that it has found a virtual environment.

    3.  **Select the Python Interpreter:**
        *   If VS Code doesn't automatically select the virtual environment, or if you want to verify or change the Python interpreter, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) to open the Command Palette.
        *   Type `Python: Select Interpreter` and press Enter.
        *   VS Code will display a list of available Python interpreters. Choose the one that is within your `.venv` virtual environment. It will typically be something like: `./.venv/bin/python` (on macOS/Linux) or `.\.venv\Scripts\python.exe` (on Windows).

    4.  **Install packages within the virtual environment:** Open the terminal in VS Code (`Ctrl+\`` or `Cmd+\``). Ensure that the virtual environment is activated in the terminal. You should see `(.venv)` at the beginning of your terminal prompt. If it's activated, you can install the project dependencies:

        ```bash
        pip install -r requirements.txt
        ```

4.  **Install dependencies:**

    Install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Review and modify `config.json`:** This is the central configuration file.
    *   Set `llm_models` with your desired Ollama models and their parameters (ensure the `name` matches your `ollama list` output).
    *   Update `question_models_to_test` and `evaluator_model_name` to point to keys within `llm_models.ollama`.
    *   Configure `files_to_test` and `file_extensions_to_test` with the manuals you want to process.
    *   Verify `question_dataset_paths` point to your JSON QA datasets (e.g., general, unanswerable).
    *   Adjust `rag_parameters` (`chunk_sizes_to_test`, `overlap_sizes_to_test`, `num_retrieved_docs`).
    *   Set the `output_dir` for saving results.
2.  **Prepare Data:**
    *   Place manuals in the `manuals/` directory.
    *   Place question/answer JSON datasets in the `question_datasets/` directory.
    *   Ensure prompt template files exist at the paths specified in `prompt_paths` (e.g., `prompt_templates/question_prompt.txt`).

## Usage Workflow

The entire testing pipeline is orchestrated by `main.py`. This single script handles creating the vector databases and then running the tests according to the settings in `config.json`.

1.  **Activate the Virtual Environment:** Before running, ensure your virtual environment is active.
    *   **On Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **On macOS and Linux:**
        ```bash
        source .venv/bin/activate
        ```

2.  **Run the Main Script:** From the `RAG2_COMPAG` directory, execute `main.py`. The script will perform all necessary steps:
    *   Initialize the embedding model.
    *   Create or update the ChromaDB vector databases for all file and parameter combinations specified in `config.json`.
    *   Run the full Question-Answering and Evaluation pipeline.
    
    ```bash
    python main.py
    ```
    The script will log its progress to the console.

## Output

Test results are saved as JSON files in the directory specified by `output_dir` in `config.json` (default: `results/`).

Filenames follow the pattern:
`{retrieval_algorithm}_{file_identifier}_{sanitized_question_model_name}_{chunk_size}_overlap_{overlap_size}_topk_{num_retrieved_docs}.json`

Each file contains:
*   `test_run_parameters`: Configuration used for that specific test run (file, models, RAG params, collection name).
*   `overall_metrics`: Calculated performance metrics (Accuracy, Precision, Recall, Specificity, F1-Score, TP/TN/FP/FN counts).
*   `timing`: Duration of the QA and Evaluation phases, and the overall duration.
*   `per_dataset_details`: Raw results broken down by the input question dataset type. Includes the original question, expected answer, model's generated answer, evaluation judgment ("yes"/"no"/"error"), and timing for that dataset.


## Try it out

To quickly test the RAG pipeline with a single question, use the `ask_question_demo.ipynb` Jupyter notebook located in the root directory.

**`ask_question_demo.ipynb` - Interactive RAG Query:**

This notebook allows you to:

1.  **Input a Question:** Interactively provide a question you want to ask.
2.  **Configure Parameters:** Easily set:
    *   The LLM to use for answering (or skip LLM answering to only see retrieved context).
    *   Whether to evaluate the LLM's answer against an expected answer you provide.
    *   The `config.json` and ChromaDB directory to use (defaults are usually fine).
3.  **Automatic Context Retrieval:** The notebook automatically identifies the relevant ChromaDB collection based on the *first* file, chunk size, and overlap size specified in your `config.json`.
4.  **View Retrieved Context:** See the text chunks retrieved from the database that are most relevant to your question.
5.  **Get LLM Answer (Optional):** If an LLM is specified, it generates an answer based on your question and the retrieved context.
6.  **Evaluate Answer (Optional):** If enabled, the LLM's answer is evaluated against an "expected answer" you provide, using another LLM for the judgment.


**How to Run:**

*   Ensure you have followed the installation steps (Python environment, dependencies, Ollama models).
*   Navigate to the `RAG2_COMPAG/` directory.
*   Open and run the `ask_question_demo.ipynb` notebook cell by cell.
*   Modify the "User Configuration" cell as needed, especially `your_question`.

This demo notebook is a great way to understand the core components of the RAG system (retrieval, LLM augmentation, evaluation) in action with minimal setup for a single query.
